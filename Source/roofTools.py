"""
roofTools.py

Utility functions for processing roof point clouds, including slicing roofs into horizontal sections
and finding the highest roof points above building corners.

Modules:
    - numpy
    - open3d
    - tqdm

Imports:
    - find_corners: Detects corners in a set of points.
    - grid_subsampling: Downsamples a point cloud using a grid/voxel approach.
    - merge_point_clouds: Merges multiple point clouds into one.
    - open_point_cloud_editor: Visualizes point clouds interactively.

Functions:
    - slice_roof_up: Slices a roof point cloud into horizontal slabs, flattens each slab, and detects corners.
    - keep_highest_point_above_corner: For each corner, finds the highest roof point directly above it.

Typical Usage:
    1. Use slice_roof_up to extract and flatten horizontal roof slices for further analysis.
    2. Use keep_highest_point_above_corner to find roof peaks above building corners.

See individual function docstrings for details.
"""
import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.spatial import cKDTree
import sys
import os

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.floorplanFinder import find_corners
from Source.pointCloudAltering import grid_subsampling, merge_point_clouds as merge_pcds
from Source.pointCloudEditor import open_point_cloud_editor as opce


def slice_roof_up(
    roof_pcd: o3d.cpu.pybind.geometry.PointCloud,
    slices_amount: int = 2,
    slab_fatness: float = 0.01,
    visualize: bool = False,
    voxel_size: float = 0.5,
    angle_threshold_deg: float = 45,
    window: int = 3,
    merge_radius: float = 0.1
) -> list[np.ndarray]:
    """
    Slice a point cloud along Z into horizontal slices and flatten each slice to its center height.

    Divides the point cloud into horizontal slices along the Z-axis, then flattens all points
    in each slice to the slice's center Z-coordinate. Useful for creating horizontal cross-sections
    of complex roof structures.

    Args:
        roof_pcd (o3d.cpu.pybind.geometry.PointCloud): Input point cloud to be sliced.
        slices_amount (int, optional): Number of horizontal slices to create. Must be at least 1. Defaults to 2.
        slab_fatness (float, optional): Half-fatness around slice center to include points.
            Points within Â±slab_fatness of slice center are included. Defaults to 0.01.
        visualize (bool, optional): Whether to visualize the slicing process. Defaults to False.
        voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.5.
        angle_threshold_deg (float, optional): Angle threshold for corner detection. Defaults to 45.
        window (int, optional): Window size for corner detection. Defaults to 3.
        merge_radius (float, optional): Merge radius for corner detection. Defaults to 0.1.

    Returns:
        list of np.ndarray: List of arrays, each containing the flattened points for a slice (ordered from low to high).
        Each array is shape (N_i, 3) for the i-th slice. Returns empty list if no points found in any slice.

    Raises:
        TypeError: If roof_pcd is not an Open3D PointCloud.
        ValueError: If slices_amount is less than 1.

    Example:
        >>> roof_pcd = create_point_cloud(roof_points, color=[0, 1, 0])
        >>> sliced_roof = slice_roof_up(roof_pcd, slices_amount=5, slab_fatness=0.02)
        >>> print(f"Created {len(sliced_roof.points)} flattened points from roof slices")
        Created 1250 flattened points from roof slices

    Note:
        - Slices are evenly distributed between minimum and maximum Z-coordinates
        - Points are flattened to their slice center height, losing original Z variation
        - Empty slices (no points within slab_fatness) are skipped
        - Uses tqdm progress bar for slice processing visualization
    """
    if not isinstance(roof_pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("roof_pcd must be an Open3D PointCloud.")

    if not isinstance(slices_amount, int):
        slices_amount = int(round(slices_amount))

    if slices_amount < 1:
        raise ValueError("slices_amount must be at least 1.")

    points = np.asarray(roof_pcd.points)
    z_vals = points[:, 2]

    z_min, z_max = z_vals.min(), z_vals.max()
    slice_centers = np.linspace(z_min, z_max, slices_amount)

    all_flattened_points = []

    for z_center in tqdm(slice_centers, desc="Processing roof slices"):
        mask = (z_vals >= z_center - slab_fatness) & (z_vals <= z_center + slab_fatness)
        slice_points = points[mask]

        if len(slice_points) == 0:
            continue

        # Flatten all points in this slice to z_center
        slice_points = slice_points.copy()  # avoid modifying original array
        slice_points[:, 2] = z_center

        # Make a temporary point cloud for the subsampling of the roof slice
        temp_pcd = o3d.cpu.pybind.geometry.PointCloud()
        temp_pcd.points = o3d.utility.Vector3dVector(slice_points)
        temp_pcd = grid_subsampling(
            temp_pcd,
            voxel_size=voxel_size,
            print_result=False  # FOR TESTING
        )

        temp_corners_array = find_corners(
            np.asarray(temp_pcd.points),
            angle_threshold_deg=angle_threshold_deg,
            window=window,
            merge_radius=merge_radius)

        temp_corners_pcd = o3d.cpu.pybind.geometry.PointCloud()
        temp_corners_pcd.points = o3d.utility.Vector3dVector(temp_corners_array)

        if visualize:
            opce(temp_corners_pcd, show_help=False)

        all_flattened_points.append(temp_corners_array)

    # Return the list of arrays, one per slice (low to high)
    return all_flattened_points


def keep_highest_point_above_corner(
    corner_pcd: o3d.cpu.pybind.geometry.PointCloud,
    full_pcd: o3d.cpu.pybind.geometry.PointCloud,
    search_radius: float = 0.01,
    compare_with_corner: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Find the highest point in full_pcd that lies approximately above each point in corner_pcd.

    For each corner point, searches for points in full_pcd within a horizontal
    distance Â±search_radius in both x and y directions and selects the point with the highest z-coordinate.

    Args:
        corner_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing corner points.
        full_pcd (o3d.cpu.pybind.geometry.PointCloud): Larger point cloud to search for points above corners.
        search_radius (float, optional): Search radius for finding points above corners. Must be greater than 0.
            Defaults to 0.01.
        compare_with_corner (bool, optional): If True, opens visualization comparing corner and highest points.
            Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud containing the highest points above each corner point,
            colored red for visualization.

    Raises:
        TypeError: If inputs are not Open3D PointCloud objects.
        ValueError: If either point cloud is empty.
        ValueError: If no points are found above any corner point.

    Example:
        >>> corner_pcd = create_point_cloud(floor_corners, color=[1, 0, 0])
        >>> ceiling_peaks = keep_highest_point_above_corner(corner_pcd, roof_pcd, search_radius=0.05)
        >>> print(f"Found {len(ceiling_peaks.points)} highest points")
        Found 4 highest points

    Note:
        - Uses rectangular search area (Â±search_radius in X and Y)
        - Returns red-colored points for visualization ðŸŸ¥.
        - With compare_with_corner=True, opens interactive visualization
        - Useful for finding roof peaks above building corners
    """

    # Validate input types
    if not all(isinstance(pc, o3d.cpu.pybind.geometry.PointCloud) for pc in [corner_pcd, full_pcd]):
        raise TypeError("Both corner_pcd and full_pcd must be Open3D PointCloud objects.")

    # Validate non-empty point clouds
    if len(corner_pcd.points) == 0 or len(full_pcd.points) == 0:
        raise ValueError("Both corner_pcd and full_pcd must contain points.")

    # Validate search radius
    if search_radius <= 0 and isinstance(search_radius, (int, float)):
        raise ValueError("search_radius must be greater than 0.")

    corner_points = np.asarray(corner_pcd.points)
    full_points = np.asarray(full_pcd.points)

    highest_points = []

    # For each corner point, find points in full_pcd close in x,y and pick the highest z
    for corner in tqdm(corner_points, desc="Finding highest points above corners", unit="corner"):
        mask = (
            (full_points[:, 0] >= corner[0] - search_radius) & (full_points[:, 0] <= corner[0] + search_radius) &
            (full_points[:, 1] >= corner[1] - search_radius) & (full_points[:, 1] <= corner[1] + search_radius)
        )
        above_points = full_points[mask]

        if above_points.size > 0:
            highest_point = above_points[np.argmax(above_points[:, 2])]
            highest_points.append(highest_point)

    if not highest_points:
        raise ValueError("No points found above the corner points.")

    # Create point cloud from highest points
    highest_pcd = o3d.cpu.pybind.geometry.PointCloud()
    highest_pcd.points = o3d.utility.Vector3dVector(highest_points)
    highest_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(highest_pcd.points), 1)))  # Red color

    if not highest_pcd.has_points():
        raise ValueError("No points found above the corner points.")

    # Placeholder for user-defined comparison logic
    if compare_with_corner:
        opce(merge_pcds([corner_pcd, highest_pcd]), show_help=False)

    return highest_pcd


def smooth_roof(
    roof_pcd: o3d.cpu.pybind.geometry.PointCloud,
    voxel_size: float = 0.5,
    upsample_factor: float = 1.0,
    visualize: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Smooth a roof point cloud to reduce spiky artifacts between layers.

    This is especially useful for wharf cellars where roofs can look arched with
    local spikes after reconstruction.

    Args:
        roof_pcd (o3d.cpu.pybind.geometry.PointCloud): Input point cloud of the roof to be smoothed.
        voxel_size (float, optional): Voxel size for downsampling. Defaults to 0.5.
        upsample_factor (float, optional): Density multiplier for visual smoothness.
            1 keeps the original point count; values >1 add interpolated points.
            Defaults to 1.0.
        visualize (bool, optional): Whether to visualize the smoothing process. Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Smoothed point cloud of the roof.
    """
    if not isinstance(roof_pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("roof_pcd must be an Open3D PointCloud.")

    if len(roof_pcd.points) == 0:
        raise ValueError("roof_pcd must contain points.")

    if not isinstance(voxel_size, (int, float)) or voxel_size <= 0:
        raise ValueError("voxel_size must be a positive number.")

    if not isinstance(upsample_factor, (int, float)) or upsample_factor < 1:
        raise ValueError("upsample_factor must be a number greater than or equal to 1.")

    original_point_count = len(roof_pcd.points)

    # Downsample first so the following 2D grid smoothing is stable and fast.
    roof_downsampled = grid_subsampling(
        roof_pcd,
        voxel_size=float(voxel_size),
        print_result=False
    )

    if len(roof_downsampled.points) == 0:
        raise ValueError("No points left after downsampling; adjust voxel_size.")

    points = np.asarray(roof_downsampled.points)
    if points.shape[0] < 3:
        return roof_downsampled

    xy = points[:, :2]
    z = points[:, 2]

    # Build an XY grid and aggregate each cell with a robust statistic (median).
    mins = xy.min(axis=0)
    cell_indices = np.floor((xy - mins) / float(voxel_size)).astype(int)
    unique_cells, inverse = np.unique(cell_indices, axis=0, return_inverse=True)

    cell_z = np.zeros(len(unique_cells), dtype=float)
    for idx in range(len(unique_cells)):
        z_vals = z[inverse == idx]
        cell_z[idx] = float(np.median(z_vals))

    cell_to_idx = {tuple(cell): idx for idx, cell in enumerate(unique_cells)}
    smoothed_cell_z = cell_z.copy()

    # Iteratively smooth cell heights while clamping isolated spikes via local MAD.
    for _ in range(2):
        new_z = smoothed_cell_z.copy()
        for idx, cell in enumerate(unique_cells):
            cx, cy = int(cell[0]), int(cell[1])
            neighbors = []
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    n_idx = cell_to_idx.get((cx + dx, cy + dy))
                    if n_idx is not None:
                        neighbors.append(smoothed_cell_z[n_idx])

            if len(neighbors) < 3:
                continue

            neighbors_arr: np.ndarray = np.asarray(neighbors, dtype=float)
            local_median = float(np.median(neighbors_arr))
            local_mean = float(np.mean(neighbors_arr))
            local_mad = float(np.median(np.abs(neighbors_arr - local_median))) + 1e-8

            if abs(smoothed_cell_z[idx] - local_median) > 2.5 * local_mad:
                new_z[idx] = local_median
            else:
                new_z[idx] = 0.6 * smoothed_cell_z[idx] + 0.4 * local_mean

        smoothed_cell_z = new_z

    smoothed_points = points.copy()
    smoothed_points[:, 2] = smoothed_cell_z[inverse]

    smoothed_roof = o3d.cpu.pybind.geometry.PointCloud()
    smoothed_roof.points = o3d.utility.Vector3dVector(smoothed_points)

    if roof_downsampled.has_colors():
        smoothed_roof.colors = roof_downsampled.colors

    if roof_downsampled.has_normals():
        smoothed_roof.normals = roof_downsampled.normals

    upsample_factor = int(upsample_factor)
    if upsample_factor > 1 and len(smoothed_points) >= 2:
        tree = cKDTree(smoothed_points)
        _, neighbor_idx = tree.query(smoothed_points, k=2)
        nearest_indices = neighbor_idx[:, 1]

        densified_points_list: list[np.ndarray] = [smoothed_points]
        densified_colors_list: list[np.ndarray] = []

        has_colors = smoothed_roof.has_colors()
        if has_colors:
            base_colors = np.asarray(smoothed_roof.colors)
            densified_colors_list.append(base_colors)

        for step in range(1, upsample_factor):
            t = step / float(upsample_factor)
            interp_points = (1.0 - t) * smoothed_points + t * smoothed_points[nearest_indices]
            densified_points_list.append(interp_points)

            if has_colors:
                interp_colors = (1.0 - t) * base_colors + t * base_colors[nearest_indices]
                densified_colors_list.append(interp_colors)

        densified_points: np.ndarray = np.vstack(densified_points_list)
        smoothed_roof.points = o3d.utility.Vector3dVector(densified_points)

        if has_colors:
            smoothed_roof.colors = o3d.utility.Vector3dVector(np.vstack(densified_colors_list))

    if visualize:
        opce(merge_pcds([roof_downsampled, smoothed_roof]), show_help=False)

    print(f"Smoothed roof point cloud: original points = {original_point_count}, "
          f"downsampled points = {len(roof_downsampled.points)}, smoothed points = {len(smoothed_roof.points)}")

    return smoothed_roof


if __name__ == "__main__":
    from Source.fileHandler import readout_LAS_file
    from Source.pointCloudAltering import alter_point_density, remove_noise_statistical
    from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
    from Source.floorplanFinder import find_boundary_from_floor, sort_points_in_hull
    from Source.wallTools import define_min_height_roof, connect_vertically_aligned_points
    from Source.pointCloudEditor import open_point_cloud_editor as opce  # noqa: F811
    from Source.pointCloudEditor import open_mesh_and_lineset_viewer as omalv  # noqa: F811
    from Source.roofTools import slice_roof_up, keep_highest_point_above_corner, smooth_roof  # noqa: F811, F401
    from Source.linesetTools import filter_lines_within_contour, contour_to_lineset

    pointCloud = readout_LAS_file("C:/Users/marcz/3D Objects/Werfkelders/xxxxxxxxxxxxxxxxxxxxxxxx.las")
    altered_pointCloud = alter_point_density(pointCloud, 1)
    cleaned_pointCloud = remove_noise_statistical(altered_pointCloud)
    pointCloudTuple = transform_pointcloud_to_height_map(cleaned_pointCloud, visualize_map_np=True, debugging_logs=True)
    floor_plan_pointCloud, ceiling_pointCloud, wall_pointCloud = pointCloudTuple
    floor_lines = find_boundary_from_floor(floor_plan_pointCloud, alpha=8, min_triangle_area=1e-10)
    floor_hull = sort_points_in_hull(floor_lines, 0.045)
    floor_hull_pcd = create_point_cloud(floor_hull, color=(1, 0, 0))
    roof_pcd, temp_wall_pcd = define_min_height_roof(ceiling_pointCloud, floor_hull_pcd, 1.5)
    print(f"Roof point cloud has {len(roof_pcd.points)} points")
    slice_amount = 10
    slab_fatness = 0.01
    voxel_size = 0.05
    angle_threshold_deg = 45
    merge_radius = 0.1

    roof_slices = slice_roof_up(
        roof_pcd,
        slices_amount=slice_amount,
        slab_fatness=slab_fatness,
        voxel_size=voxel_size,
        angle_threshold_deg=angle_threshold_deg,
        merge_radius=merge_radius
    )

    opce(
        merge_pcds(
            [
                create_point_cloud(
                    slice_points,
                    color=(0, 1, 0)
                ) for slice_points in roof_slices
            ]
        ),
        show_help=False
    )

    xy_tolerance = 0.1
    max_line_length = 0.5

    roof_lineset = o3d.geometry.LineSet()
    for i in range(len(floor_hull) - 1, 0, -1):
        print(f"Slice {i + 1}: {len(roof_slices[i])} points")
        roof_lineset += connect_vertically_aligned_points(
            roof_slices[i - 1],
            roof_slices[i],
            float(xy_tolerance)
        )
        roof_lineset = contour_to_lineset(
            sort_points_in_hull(floor_hull[i]),
            max_line_length=max_line_length
        )

    roof_lineset = filter_lines_within_contour(floor_hull, roof_lineset)

    omalv(roof_lineset)
