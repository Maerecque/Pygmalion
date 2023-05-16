import open3d as o3d
import numpy as np


def grid_subsampling(pcd: o3d.cpu.pybind.geometry.PointCloud, voxelSize: float = 0.05) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to normalize the points in a point cloud over a grid.
    So I found out, maybe the hard way. This method will always normalize the size of the point cloud based on how many points are in the cloud.
    A large point cloud will not be decimated by the same size as a smaller point cloud. It is all dependent on the original density of the cloud.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud to be normalized.

        voxelSize (float, optional): Distance between points that is allowed.
            Defaults to 0.05.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Down sampled point cloud with normalized point positions.
    """
    # Downsample the point cloud to a regular grid using voxel_down_sample
    downsampled_pcd = pcd.voxel_down_sample(voxelSize)
    print(f'The point cloud has been resized after grid normalization from {len(pcd.points):,} to {len(downsampled_pcd.points):,}')

    return downsampled_pcd


def remove_noise_statistical(
    inputPointCloud: o3d.cpu.pybind.geometry.PointCloud,
    showRemovedPoints: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    print_removal_amount: bool = True,
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to remove noise from a point cloud. This removes points that are further away from their neighbors in average.
    !!! This function is still an experimental feature. !!!
    This function seems to perform much better on removing noise without altering any hyperparameters from both mobile and static scans.

    Args:
        inputPointCloud (open3d.cpu.pybind.geometry.PointCloud): A point cloud where the noise will be removed from.

        showRemovedPoints (bool, optional): Boolean to show an example with the removed points from the cloud marked in red.
            Defaults to False.

        nb_neighbors (int, optional): Number of neighbors around the target point.
            Defaults to 20.

        std_ratio (float, optional): Standard deviation ratio.
            Defaults to 2.0.

        print_removal_amount (bool, optional):
            Boolean to print the amount of points that were removed.
            Defaults to True.

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: A cleaned up version of the point cloud.
    """
    cl, ind = inputPointCloud.remove_statistical_outlier(nb_neighbors, std_ratio)

    amount_removed_points = len(inputPointCloud.points) - len(cl.points)
    if amount_removed_points < 1:
        amount_removed_points = "No"

    if len(inputPointCloud.points) == len(cl.points):
        amount_removed_points = f"All the ({len(cl.points)})"

    if print_removal_amount: print(str(amount_removed_points) + " points were removed as outliers.")

    # So there is a bug with this if statement.
    # For some unknown reason, the outlier_cloud will not be coloured red. This wil happen randomly with the same settings in the same point cloud.
    if showRemovedPoints:
        outlier_cloud = inputPointCloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([outlier_cloud, inputPointCloud], left=0, top=45, window_name="Remove noise with statistical")

    return cl


def remove_noise_radius(
    inputPointCloud: o3d.cpu.pybind.geometry.PointCloud,
    showRemovedPoints: bool = False,
    nb_points: int = 10,
    radius: float = 0.1
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to remove noise from a point cloud. This removes points that have neighbors less than nb_points in a sphere of a given radius.
    !!! This function is still an experimental feature. !!!

    Args:
        inputPointCloud (open3d.cpu.pybind.geometry.PointCloud): A point cloud where the noise will be removed from.

        showRemovedPoints (bool, optional): Boolean to show an example with the removed points from the cloud marked in red.
            Defaults to False.

        nb_points (int, optional): Number of points within the radius.
            Defaults to 10.

        radius (float, optional): Radius of the sphere.
            Defaults to 0.1.

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: A cleaned up version of the point cloud.
    """
    cl, ind = inputPointCloud.remove_radius_outlier(nb_points, radius)

    amount_removed_points = len(inputPointCloud.points) - len(cl.points)
    if amount_removed_points < 1:
        amount_removed_points = "No"

    if len(inputPointCloud.points) == len(cl.points):
        amount_removed_points = f"All the ({len(cl.points)})"

    print(str(amount_removed_points) + " points were removed as outliers.")

    if showRemovedPoints:
        outlier_cloud = inputPointCloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([outlier_cloud, inputPointCloud], left=0, top=45, window_name="Remove noise with radius")

    return cl


def combine_point_cloud(
    inputPointCloud: o3d.cpu.pybind.geometry.PointCloud,
    classifiedPointCloud: o3d.cpu.pybind.geometry.PointCloud
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function made to remove any points that were found in the DBScan from the original scan.
    This is made mainly for visualizing the results of the remaining points.

    Args:
        inputPointCloud (o3d.cpu.pybind.geometry.PointCloud): Pointcloud with the base scan in it.

        classifiedPointCloud (o3d.cpu.pybind.geometry.PointCloud): Pointcloud after the classification with points to be removed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Pointcloud without any points that were in the given classification pointcloud.
    """
    distances = inputPointCloud.compute_point_cloud_distance(classifiedPointCloud)
    distances_np = np.asarray(distances)
    zero_distance_mask = (distances_np == 0)
    pcd1_without_duplicates = inputPointCloud.select_by_index(np.where(zero_distance_mask == False)[0])
    return pcd1_without_duplicates
