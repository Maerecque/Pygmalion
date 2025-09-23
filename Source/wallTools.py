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


def extract_wall_points(
    tbp_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_contour_pcd: o3d.cpu.pybind.geometry.PointCloud,
    search_radius: float = 0.025,
    print_bool: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Keep all tbp_pcd points that are above the floor contour.

    For each floor contour point this collects every TBP point within `search_radius`
    in XY whose Z is >= that floor contour point's Z. The result is the union of
    all such TBP points (deduplicated), preserving all points that lie above the
    contour rather than producing a single representative point per contour point.

    Args:
        tbp_pcd, floor_contour_pcd: Open3D PointClouds
        search_radius: XY search radius to associate tbp points to floor points
        other args kept for API compatibility

    Returns:
        Open3D PointCloud of selected points (colored blue)
    """
    if not all(isinstance(pc, o3d.cpu.pybind.geometry.PointCloud) for pc in [tbp_pcd, floor_contour_pcd]):
        raise TypeError("Both tbp_pcd and floor_contour_pcd must be Open3D PointCloud objects.")
    if len(tbp_pcd.points) == 0 or len(floor_contour_pcd.points) == 0:
        raise ValueError("Both tbp_pcd and floor_contour_pcd must contain points.")
    if search_radius <= 0:
        raise ValueError("search_radius must be a positive float.")

    floor_points = np.asarray(floor_contour_pcd.points)
    tbp_all_points = np.asarray(tbp_pcd.points)

    if print_bool:
        print(f"Number of floor points: {floor_points.shape[0]}")
        print(f"Number of TBP points: {tbp_all_points.shape[0]}")

    floor_xy = floor_points[:, :2]
    floor_z = floor_points[:, 2]
    tbp_xy = tbp_all_points[:, :2]
    tbp_z = tbp_all_points[:, 2]

    if floor_xy.shape[0] == 0 or tbp_xy.shape[0] == 0:
        raise ValueError("No points available after conversion to arrays.")

    # Build KDTree on TBP XY so we can query all TBP points around each floor point
    tbp_tree = cKDTree(tbp_xy, leafsize=2)

    selected_indices = set()

    # For each floor point, get all tbp indices within search_radius and keep those above the floor z
    for i, (fxy, fz) in enumerate(zip(floor_xy, floor_z)):
        idxs = tbp_tree.query_ball_point(fxy, r=search_radius)
        if not idxs:
            continue
        idxs = np.asarray(idxs, dtype=int)
        # keep tbp points whose z is >= floor z at this contour point
        above_mask = tbp_z[idxs] >= fz
        if np.any(above_mask):
            for idx in idxs[above_mask]:
                selected_indices.add(int(idx))

    if len(selected_indices) == 0:
        raise ValueError("No TBP points found above the contour within the given search radius.")

    # Create ordered array of unique selected points
    selected_idx_list = np.fromiter(sorted(selected_indices), dtype=int)
    kept_points = tbp_all_points[selected_idx_list]

    if print_bool:
        print(f"Selected {kept_points.shape[0]} TBP points above the contour (union over all floor points).")

    result_pcd = create_point_cloud(kept_points, [0, 0, 1])
    return result_pcd


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
