import numpy as np
import open3d as o3d
import os, sys  # noqa: E401
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
import Source.pointCloudEditor as pce
import Source.fileHandler as fh


def grid_subsampling(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    voxel_size: float = 0.05,
    print_result: bool = True
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to normalize the points in a point cloud over a grid.
    So I found out, maybe the hard way. This method will always normalize the size of the point cloud based on how many points
    are in the cloud. A large point cloud will not be decimated by the same size as a smaller point cloud. It is all dependent
    on the original density of the cloud.
    NOTE:   Should research a more reliable way to normalize the point cloud,
            find a way to normalize based on original point distance.
            `voxel_down_sample_and_trace` exists.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud to be normalized.

        voxel_size (float, optional): Distance between points that is allowed.
            Defaults to 0.05.

        print_result (bool, optional): Whether to print the result. Defaults to True.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Down sampled point cloud with normalized point positions.
    """
    # Downsample the point cloud to a regular grid using voxel_down_sample
    downsampled_pcd = pcd.voxel_down_sample(voxel_size)
    if print_result:
        print(f'The point cloud has been resized after grid normalization from {len(pcd.points):,} to {len(downsampled_pcd.points):,}')  # noqa: E501

    return downsampled_pcd


def remove_noise_statistical(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    show_removed_points: bool = False,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
    print_removal_amount: bool = True,
) -> o3d.cpu.pybind.geometry.PointCloud:
    """Remove noise from a point cloud using statistical outlier removal.

    Args:
        input_pcd (open3d.cpu.pybind.geometry.PointCloud): Point cloud to clean.

        show_removed_points (bool, optional): Show removed points marked in red. Defaults to False.

        nb_neighbors (int, optional): Number of neighbors to analyze for each point. Defaults to 20.

        std_ratio (float, optional): Standard deviation multiplier for thresholding. Defaults to 2.0.

        print_removal_amount (bool, optional): Print how many points were removed. Defaults to True.

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: Cleaned point cloud.
    """
    cl, ind = input_pcd.remove_statistical_outlier(nb_neighbors, std_ratio)

    total_points = len(input_pcd.points)
    kept_points = len(cl.points)
    removed_points = total_points - kept_points

    if print_removal_amount:
        if removed_points == 0:
            print("No points were removed as outliers.")
        elif removed_points == total_points:
            print(f"All the points ({total_points}) were removed as outliers.")
        else:
            print(f"{removed_points} points were removed as outliers.")

    if show_removed_points:
        outlier_cloud = input_pcd.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])  # Mark outliers in red

        inlier_cloud_ex = get_difference_point_cloud(input_pcd, outlier_cloud)
        o3d.visualization.draw_geometries(
            [inlier_cloud_ex, outlier_cloud],
            left=0,
            top=45,
            window_name="Remove noise with statistical"
        )

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


def merge_point_clouds(pcd_list: list[o3d.cpu.pybind.geometry.PointCloud]) -> o3d.cpu.pybind.geometry.PointCloud:
    """Merges a list of point clouds into a single point cloud.
    This function assumes that all point clouds in the list have the same structure (i.e., they all have points and colors).
    The function does not check for duplicates or overlapping points.
    The first point cloud that is passed will overrule overlapping points.

    Args:
        pcd_list (list[o3d.cpu.pybind.geometry.PointCloud]): List of point clouds to merge.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Merged point cloud containing all points and colors from the input list.
    """
    # Check if all point clouds are actually point clouds and not ndarrays, if they are convert them to point clouds
    pcd_list = [o3d.geometry.PointCloud(points=o3d.utility.Vector3dVector(pcd)) if isinstance(pcd, np.ndarray) else pcd for pcd in pcd_list]  # noqa: E501

    # Check that all point clouds have points if one misses points, drop it from the list
    pcd_list = [pcd for pcd in pcd_list if len(pcd.points) > 0]
    if not pcd_list:
        raise ValueError("All point clouds must have points.")

    # Check if all point clouds have colour, if one misses color, make it grey
    for pcd in pcd_list:
        if not pcd.has_colors():
            pcd.colors = o3d.utility.Vector3dVector(np.tile([0.5, 0.5, 0.5], (len(pcd.points), 1)))

    merged_pcd = o3d.geometry.PointCloud()
    merged_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(pcd.points) for pcd in pcd_list])
    )
    merged_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([np.asarray(pcd.colors) for pcd in pcd_list])
    )
    return merged_pcd


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
