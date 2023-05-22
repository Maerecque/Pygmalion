import open3d as o3d
import numpy as np

import fileHandler as fh
import pointCloudEditor as pce


def grid_subsampling(pcd: o3d.cpu.pybind.geometry.PointCloud, voxel_size: float = 0.05) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to normalize the points in a point cloud over a grid.
    So I found out, maybe the hard way. This method will always normalize the size of the point cloud based on how many points are in the cloud.
    A large point cloud will not be decimated by the same size as a smaller point cloud. It is all dependent on the original density of the cloud.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud to be normalized.

        voxel_size (float, optional): Distance between points that is allowed.
            Defaults to 0.05.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Down sampled point cloud with normalized point positions.
    """
    # Downsample the point cloud to a regular grid using voxel_down_sample
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    print(f'The point cloud has been resized after grid normalization from {len(pcd.points):,} to {len(downsampled_pcd.points):,}')

    return downsampled_pcd


def remove_noise_statistical(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    show_removed_points: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    print_removal_amount: bool = True,
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to remove noise from a point cloud. This removes points that are further away from their neighbors in average.
    !!! This function is still an experimental feature. !!!
    This function seems to perform much better on removing noise without altering any hyperparameters from both mobile and static scans.

    Args:
        input_pcd (open3d.cpu.pybind.geometry.PointCloud): A point cloud where the noise will be removed from.

        show_removed_points (bool, optional): Boolean to show an example with the removed points from the cloud marked in red.
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
    cl, ind = input_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    amount_removed_points = len(input_pcd.points) - len(cl.points)
    if amount_removed_points < 1:
        amount_removed_points = "No"

    if len(input_pcd.points) == len(cl.points):
        amount_removed_points = f"All the ({len(cl.points)})"

    if print_removal_amount: print(str(amount_removed_points) + " points were removed as outliers.")

    # So there is a bug with this if statement.
    # For some unknown reason, the outlier_cloud will not be coloured red. This wil happen randomly with the same settings in the same point cloud.
    if show_removed_points:
        outlier_cloud = input_pcd.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        inlier_cloud_ex = get_difference_point_cloud(input_pcd, outlier_cloud)
        o3d.visualization.draw_geometries([inlier_cloud_ex, outlier_cloud], left=0, top=45, window_name="Remove noise with statistical")
    return cl


def combine_point_cloud(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    classified_pcd: o3d.cpu.pybind.geometry.PointCloud
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function made to remove any points that were found in the DBScan from the original scan.
    This is made mainly for visualizing the results of the remaining points.

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): Pointcloud with the base scan in it.

        classified_pcd (o3d.cpu.pybind.geometry.PointCloud): Pointcloud after the classification with points to be removed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Pointcloud without any points that were in the given classification pointcloud.
    """
    distances = input_pcd.compute_point_cloud_distance(classified_pcd)
    distances_np = np.asarray(distances)
    zero_distance_mask = (distances_np == 0)
    pcd1_without_duplicates = input_pcd.select_by_index(np.where(zero_distance_mask == False)[0])
    return pcd1_without_duplicates


def get_difference_point_cloud(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    classified_pcd: o3d.cpu.pybind.geometry.PointCloud
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function made to keep any points that were not found in the DBScan from the original scan.
    This is made mainly for visualizing the results of the non-remaining points.

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): Pointcloud with the base scan in it.

        classified_pcd (o3d.cpu.pybind.geometry.PointCloud): Pointcloud after the classification with points to be removed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Pointcloud without any points that were in the given classification pointcloud.
    """
    distances = input_pcd.compute_point_cloud_distance(classified_pcd)
    distances_np = np.asarray(distances)
    non_zero_distance_mask = (distances_np != 0)
    pcd1_without_duplicates = input_pcd.select_by_index(np.where(non_zero_distance_mask)[0])
    return pcd1_without_duplicates


if __name__ == "__main__":
    # Get the path to the LAS/LAZ file
    filename = fh.get_file_path("Select a LAS or LAZ file", ["*.las", "*.laz"])

    # Read the LAS/LAZ file
    pointcloud = fh.readout_LAS_file(filename)

    # Other pointclouds
    filename2 = fh.get_file_path("Select a LAS or LAZ file", ["*.las", "*.laz"])

    # Read the LAS/LAZ file
    pointcloud2 = fh.readout_LAS_file(filename2)

    new_pcd = get_difference_point_cloud(pointcloud, pointcloud2)

    pce.open_point_cloud_editor(new_pcd)
