"""
wallTools.py

Wall and height-based point cloud slicing utilities for architectural 3D geometry using Open3D and NumPy.

Modules:
    - numpy
    - open3d
    - scipy.spatial
    - typing

Imports:
    - create_point_cloud: Utility for creating colored Open3D point clouds.
    - cKDTree: Fast spatial search for point matching.

Functions:
    - create_correct_height_slice: Generates a horizontal slice of a point cloud at a specified height above a reference floor
      contour, with outlier filtering and neighbor checks.
    - keep_wall_points_from_x_height: Filters wall point cloud to retain only points above a given height relative to the minimum
      floor level.
    - connect_vertically_aligned_points: Connects pairs of floor and wall points whose XY coordinates match within a tolerance,
      returning an Open3D LineSet.
    - connect_vertically_aligned_points2: Connects base-level points to one or more sets of upper-level points by XY proximity,
      supporting multi-level connections and returning a LineSet.

Typical Usage:
    1. Use create_correct_height_slice to extract a wall slice at a desired height above the floor.
    2. Use keep_wall_points_from_x_height to remove points below a threshold, e.g., to exclude furniture.
    3. Use connect_vertically_aligned_points or connect_vertically_aligned_points2 to visualize vertical relationships between
       floor and wall slices or between multiple levels.

See individual function docstrings for details.
"""

import open3d as o3d
import numpy as np
from typing import Union, List
from scipy.spatial import cKDTree

from Source.heightMapModule import create_point_cloud


def create_correct_height_slice(
    tbp_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_contour_pcd: o3d.cpu.pybind.geometry.PointCloud,
    height: float = 1.5,
    search_radius: float = 0.025,
    height_tol: float = 0.75,
    neighbor_window: int = 4,
    min_low_neighbors: int = 3,
    print_bool: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Create a slice of the point cloud at a specified height above the floor.

    For each floor contour point, finds the closest point in the target point cloud
    within the search radius that is at or below the slice height, creating a horizontal
    slice at the specified height above the floor reference level.

    Args:
        tbp_pcd (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be sliced.
        floor_contour_pcd (o3d.cpu.pybind.geometry.PointCloud): The floor point cloud to reference for height.
        height (float, optional): The height at which to slice the point cloud in meters. Defaults to 1.5.
        search_radius (float, optional): The radius within which to search for points in the tbp_pcd.
            Must be a positive float. Defaults to 0.025.
        height_tol (float, optional): The height tolerance in meters. Defaults to 0.75.
        neighbor_window (int, optional): The number of neighbors on each side to check. Defaults to 4.
        min_low_neighbors (int, optional): If at least this many neighbors are also low, keep the point. Defaults to 3.
        print_bool (bool, optional): Whether to print progress information. Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: A new point cloud containing the sliced points at the target height.

    Raises:
        TypeError: If either input is not an Open3D PointCloud object.
        ValueError: If either point cloud is empty.
        ValueError: If search_radius is not positive.

    Example:
        >>> # Create a wall slice 1.5m above floor level
        >>> floor_pcd = create_point_cloud(floor_corners, color=[1, 0, 0])
        >>> wall_slice = create_correct_height_slice(building_pcd, floor_pcd, height=1.5)
        >>> print(f"Created slice with {len(wall_slice.points)} points")
        Created slice with 24 points

    Note:
        - Uses floor minimum Z-coordinate as reference height
        - For points above slice height, creates new points at the slice level
        - Prints progress information during processing
        - Returns blue-colored point cloud for visualization 🟦.
    """
    if not all(isinstance(pc, o3d.cpu.pybind.geometry.PointCloud) for pc in [tbp_pcd, floor_contour_pcd]):
        raise TypeError("Both tbp_pcd and floor_contour_pcd must be Open3D PointCloud objects.")

    if len(tbp_pcd.points) == 0 or len(floor_contour_pcd.points) == 0:
        raise ValueError("Both tbp_pcd and floor_contour_pcd must contain points.")

    if search_radius <= 0:
        raise ValueError("search_radius must be a positive float.")

    # Find at what height the wall points should be
    floor_height = np.min(np.asarray(floor_contour_pcd.points)[:, 2])
    slice_height = floor_height + height

    tbp_countour_points = []
    floor_points = np.asarray(floor_contour_pcd.points)
    tbp_all_points = np.asarray(tbp_pcd.points)

    if print_bool:
        # Show amount of floor points
        print(f"Number of floor points: {floor_points.shape[0]}")
        print(f"Number of TBP points: {tbp_all_points.shape[0]}")

    for floor_pt in floor_points:
        mask = (
            (np.abs(tbp_all_points[:, 0] - floor_pt[0]) <= (search_radius)) &  # X-axis tolerance
            (np.abs(tbp_all_points[:, 1] - floor_pt[1]) <= (search_radius))   # Y-axis tolerance
        )
        candidates = tbp_all_points[mask]
        if candidates.size > 0:
            if print_bool:
                print(f"Found {len(candidates)} candidates for floor point {floor_pt}.")
            # Pick the closest point that is not above the slice height but as close as possible
            below_slice = candidates[candidates[:, 2] <= slice_height]
            if below_slice.size > 0:
                if print_bool:
                    print(f"Found {len(below_slice)} candidates below the slice height for floor point {floor_pt}.")

                idx = np.argmin(np.abs(below_slice[:, 2] - slice_height))

                # If there are more than one point near the floor point, pick the one with the highest Z value
                if below_slice.shape[0] > 1:
                    idx = np.argmax(below_slice[:, 2])

                tbp_countour_points.append(below_slice[idx])
            else:
                # If there are no points at or below the slice height the ceiling should be above the slice height
                # Which would mean you can place a point at the slice height
                new_point = np.array([floor_pt[0], floor_pt[1], slice_height])
                tbp_countour_points.append(new_point)

    tbp_countour_points = np.array(tbp_countour_points)

    if len(tbp_countour_points) > 2 * neighbor_window:
        keep_mask = np.ones(len(tbp_countour_points), dtype=bool)
        for i, pt in enumerate(tbp_countour_points):
            if pt[2] < slice_height - height_tol:
                # Check neighbors
                start = max(0, i - neighbor_window)
                end = min(len(tbp_countour_points), i + neighbor_window + 1)
                neighbors = np.delete(tbp_countour_points[start:end], i - start, axis=0)
                low_neighbors = np.sum(neighbors[:, 2] < slice_height - height_tol)
                if low_neighbors < min_low_neighbors:
                    keep_mask[i] = False  # Outlier, remove
        tbp_countour_points = tbp_countour_points[keep_mask]

    auspuf = create_point_cloud(tbp_countour_points, [0, 0, 1])

    return auspuf


def keep_wall_points_from_x_height(
    wall_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_pcd: o3d.cpu.pybind.geometry.PointCloud,
    height: float = 1.5
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Keep wall points above a certain height relative to the floor.

    Filters the wall point cloud to retain only points that are above a specified
    height threshold relative to the minimum floor height.

    Args:
        wall_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing wall points.
        floor_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing floor points for height reference.
        height (float, optional): Height above the floor to keep wall points. Defaults to 1.5.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud containing wall points above the specified height,
                                           colored blue for visualization.

    Raises:
        TypeError: If either input is not an Open3D PointCloud object.
        ValueError: If either point cloud is empty.
        ValueError: If no wall points are found above the specified height.

    Example:
        >>> floor_pcd = create_point_cloud(floor_points, color=[1, 0, 0])
        >>> filtered_walls = keep_wall_points_from_x_height(wall_pcd, floor_pcd, height=2.0)
        >>> print(f"Kept {len(filtered_walls.points)} wall points above 2.0m")
        Kept 1523 wall points above 2.0m

    Note:
        - Uses minimum Z-coordinate from floor_pcd as reference height
        - All filtered points are colored blue for visualization
        - Useful for removing furniture and lower structural elements
    """
    if not isinstance(
        wall_pcd, o3d.cpu.pybind.geometry.PointCloud
    ) or not isinstance(
        floor_pcd, o3d.cpu.pybind.geometry.PointCloud
    ):
        raise TypeError("Both wall_pcd and floor_pcd must be Open3D PointCloud objects.")
    if len(wall_pcd.points) == 0 or len(floor_pcd.points) == 0:
        raise ValueError("Both ceiling_pcd and floor_pcd must contain points.")
    # Get the minimum z value from the floor point cloud
    floor_height = np.min(np.asarray(floor_pcd.points)[:, 2])

    # Calculate the minimum z value for the ceiling points
    min_z_value = floor_height + height

    # Filter the wall point cloud to keep only points above the minimum z value
    wall_points = np.asarray(wall_pcd.points)
    wall_points_filtered = wall_points[wall_points[:, 2] > min_z_value]

    if wall_points_filtered.size == 0:
        raise ValueError("No wall points found above the specified height.")

    # Create a new point cloud with the filtered points
    filtered_wall_pcd = o3d.cpu.pybind.geometry.PointCloud()
    filtered_wall_pcd.points = o3d.utility.Vector3dVector(wall_points_filtered)
    # Make all new points blue
    filtered_wall_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(filtered_wall_pcd.points), 1)))

    return filtered_wall_pcd


def connect_vertically_aligned_points(
    floor_points: np.ndarray,
    wall_points: np.ndarray,
    xy_tol: float = 1e-2
) -> o3d.geometry.LineSet:
    """
    Connects pairs of 3D points from two sets (floor and wall) if their XY coordinates are aligned within a specified tolerance.
    For each point in `floor_points`, finds the closest point in `wall_points` (in XY plane) within `xy_tol` distance,
    and creates a line connecting them. Useful for visualizing vertical connections between lower and upper surfaces.

    Args:
        floor_points (np.ndarray): Array of shape (N, 3) containing 3D coordinates of floor (lower) points.
        wall_points (np.ndarray): Array of shape (M, 3) containing 3D coordinates of wall (upper) points.
        xy_tol (float, optional): Maximum allowed distance in XY plane for points to be considered vertically aligned.
            Defaults to 1e-2.

    Returns:
        o3d.geometry.LineSet: Open3D LineSet object containing all points and lines connecting vertically aligned pairs.

    Example:
        >>> floor = np.array([[0, 0, 0], [1, 1, 0]])
        >>> wall = np.array([[0, 0, 3], [1, 1, 3]])
        >>> lineset = connect_vertically_aligned_points(floor, wall, xy_tol=0.05)
        >>> print(len(lineset.lines))  # Should print 2

    Note:
        - Uses KDTree for efficient nearest neighbor search in the XY plane.
        - Only connects each floor point to its closest wall point if within tolerance.
        - All points (floor and wall) are included in the output LineSet.
    """
    # Always convert to numpy array for slicing
    floor_points = np.asarray(floor_points)
    wall_points = np.asarray(wall_points)
    floor_xy = floor_points[:, :2]
    wall_xy = wall_points[:, :2]
    lines = []
    all_points = np.vstack([floor_points, wall_points])
    wall_offset = len(floor_points)

    # Build KDTree for fast lookup
    from scipy.spatial import cKDTree
    wall_tree = cKDTree(wall_xy, leafsize=2)
    for i, xy in enumerate(floor_xy):
        dist, idx = wall_tree.query(xy, k=1)
        if dist <= xy_tol:
            lines.append([i, wall_offset + idx])

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def connect_vertically_aligned_points2(
    base_level_points: np.ndarray,
    upper_level_points: Union[np.ndarray, List[np.ndarray]],
    xy_tol: float = 1e-2
) -> o3d.geometry.LineSet:
    """
    Connects pairs of 3D points from two sets (base and upper levels) if their XY coordinates are aligned within a specified
    tolerance. For each point in `base_level_points`, finds the closest point in each set of `upper_level_points` (in XY plane)
    within `xy_tol` distance, and creates a line connecting them. Useful for visualizing vertical connections between lower and
    upper surfaces.

    Args:
        base_level_points (np.ndarray): Array of shape (N, 3) containing 3D coordinates of base (lower) points.
        upper_level_points (Union[np.ndarray, List[np.ndarray]]): Either a single Mx3 array or a list of arrays, each
        representing an upper level.
        xy_tol (float, optional): Maximum allowed distance in XY plane for points to be considered vertically aligned.
        Defaults to 1e-2.

    Returns:
        o3d.geometry.LineSet: Open3D LineSet object containing all points and lines connecting vertically aligned pairs.

    Example:
        >>> base = np.array([[0, 0, 0], [1, 1, 0]])
        >>> upper1 = np.array([[0, 0, 3], [1, 1, 3]])
        >>> upper2 = np.array([[0, 0, 6], [1, 1, 6]])
        >>> lineset = connect_vertically_aligned_points2(base, [upper1, upper2], xy_tol=0.05)
        >>> print(len(lineset.lines))  # Should print 2

    Note:
        - Uses KDTree for efficient nearest neighbor search in the XY plane.
        - Only connects each base point to its closest upper point if within tolerance.
        - All points (base and upper levels) are included in the output LineSet.
    """
    base_level_points = np.asarray(base_level_points)

    # Normalize: always work with a list of ndarrays
    if isinstance(upper_level_points, np.ndarray):
        upper_level_points = [upper_level_points]

    # Keep track of all points and line connections
    all_points = [base_level_points]
    lines = []
    point_offset = len(base_level_points)  # noqa: F841

    # Flatten list of all upper points for Open3D
    for lvl in upper_level_points:
        all_points.append(np.asarray(lvl))

    all_points = np.vstack(all_points)

    # For each base point, check levels in order until match is found
    for i, xy in enumerate(base_level_points[:, :2]):
        matched = False
        offset = len(base_level_points)  # start after base points
        for lvl in upper_level_points:
            lvl = np.asarray(lvl)
            if lvl.shape[0] == 0:
                offset += 0
                continue

            lvl_xy = lvl[:, :2]
            tree = cKDTree(lvl_xy, leafsize=2)

            dist, idx = tree.query(xy, k=1)
            if dist <= xy_tol:
                # Connect base point i with the found point in this level
                lines.append([i, offset + idx])
                matched = True  # noqa: F841
                break  # stop checking this base point once matched

            offset += len(lvl)

        # if not matched, do nothing → base point stays unconnected

    # Build LineSet
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(all_points)
    lineset.lines = o3d.utility.Vector2iVector(lines)

    return lineset
