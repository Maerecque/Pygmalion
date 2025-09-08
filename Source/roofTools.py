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
            Points within ±slab_fatness of slice center are included. Defaults to 0.01.
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

    for z_center in tqdm(slice_centers, desc="Processing slices"):
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
    distance ±search_radius in both x and y directions and selects the point with the highest z-coordinate.

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
        - Uses rectangular search area (±search_radius in X and Y)
        - Returns red-colored points for visualization 🟥.
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
    highest_pcd.colors = o3d.utility.Vector3dVector(np.tile([1, 0, 0], (len(highest_pcd.points), 1)))  # red color

    if not highest_pcd.has_points():
        raise ValueError("No points found above the corner points.")

    # Placeholder for user-defined comparison logic
    if compare_with_corner:
        opce(merge_pcds([corner_pcd, highest_pcd]), show_help=False)

    return highest_pcd
