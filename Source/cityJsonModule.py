import open3d as o3d
import numpy as np
from tqdm import tqdm
import sys
import os
from typing import List, Union

import json
import tkinter as tk
from tkinter import filedialog


# Use alpha shape to find concave boundary (captures inner/underlying edges)
from shapely.geometry import MultiPoint, LineString, Polygon
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree

# # Add this line with your other imports (around line 11)
# from temp import build_lod2_from_floor_and_pointcloud

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.fileHandler import get_file_path, readout_LAS_file
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.pointCloudEditor import open_point_cloud_editor as opce
from Source.pointCloudAltering import remove_noise_statistical as rns, merge_point_clouds as merge_pcds, grid_subsampling


def load_and_preprocess_pointcloud() -> o3d.geometry.PointCloud:
    """
    Load a point cloud file selected by the user through a file dialog.

    Prompts the user to select a LAS or LAZ point cloud file using a file dialog,
    then loads the file using the custom file handler. No preprocessing is applied.

    Returns:
        o3d.geometry.PointCloud: The loaded point cloud from the selected file.
                                Returns None if no file is selected or if loading fails.

    Raises:
        FileNotFoundError: If the selected file does not exist or cannot be accessed.
        ValueError: If the selected file is not a valid LAS/LAZ format.
        IOError: If there are issues reading the file (corrupted data, permissions, etc.).

    Example:
        >>> pcd = load_and_preprocess_pointcloud()
        Point cloud loaded and preprocessed. Now starting surface reconstruction...
        >>> print(f"Loaded point cloud with {len(pcd.points)} points")
        Loaded point cloud with 125000 points

    Note:
        - Only LAS and LAZ file formats are supported
        - Despite the function name suggesting preprocessing, no actual preprocessing is performed
        - The user can cancel the file dialog, in which case None may be returned
    """
    pcd = readout_LAS_file(get_file_path("Select a point cloud file to process", "LAS files (*.las *.laz)"))
    print("Point cloud loaded and preprocessed. Now starting surface reconstruction...")
    return pcd


def find_boundary_from_floor(pcd: o3d.geometry.PointCloud, alpha: float = 10, min_triangle_area: float = 1e-10) -> np.ndarray:
    """
    Find boundary/contour points in a 3D point cloud by projecting to 2D and detecting alpha shape boundaries.

    Projects a 3D point cloud onto the XY plane, computes an alpha shape (concave hull)
    to detect boundaries, and returns the 3D coordinates of points on the detected boundary.

    Args:
        pcd (o3d.geometry.PointCloud): The input 3D point cloud containing structural data.
        alpha (float, optional): Alpha parameter controlling boundary concavity.
            Lower values (1-5) create tighter boundaries, higher values (10-20) create smoother ones. Defaults to 10.
        min_triangle_area (float, optional): Minimum area threshold for triangles in Delaunay triangulation.
            Defaults to 1e-10.

    Returns:
        np.ndarray: Array of shape (N, 3) containing 3D coordinates of detected boundary points.
                   Returns None if an error occurs.

    Raises:
        TypeError: If input is not a valid Open3D PointCloud object.
        ValueError: If the point cloud is empty or contains no points.

    Example:
        >>> building_points = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        >>> pcd = o3d.geometry.PointCloud()
        >>> pcd.points = o3d.utility.Vector3dVector(building_points)
        >>> boundary_points = find_lines_in_pointcloud(pcd, alpha=8.0)
        >>> print(f"Detected {len(boundary_points)} boundary points")

    Note:
        - Projects all points to XY plane (Z coordinates preserved in output)
        - Works best with point clouds having clear structural boundaries
        - For noisy data, consider preprocessing with noise removal
    """
    try:
        if not isinstance(pcd, o3d.geometry.PointCloud):
            raise TypeError("Input must be an Open3D PointCloud object.")
        if len(pcd.points) == 0:
            raise ValueError("Point cloud is empty. Please provide a valid point cloud with points.")
        if type(pcd.points) is not o3d.utility.Vector3dVector:
            raise TypeError("Point cloud points must be of type Vector3dVector.")

        # Convert the point cloud to a numpy array
        points = np.asarray(pcd.points)

        # Project to 2D (XY plane) for boundary detection
        points_2d = points[:, :2]

        # Choose alpha parameter (smaller = tighter fit, adjust as needed)
        boundary = alpha_shape(points_2d, alpha, min_triangle_area=1e-10)

        # Get boundary coordinates as indices in the original array
        if boundary.geom_type == 'Polygon':
            boundary_coords = np.array(boundary.exterior.coords)
        else:
            boundary_coords = np.array(boundary.convex_hull.exterior.coords)

        # Find the indices of the boundary points in the original array
        tree = cKDTree(points_2d)
        _, idx = tree.query(boundary_coords, k=1)
        line_points = points[idx]

        return line_points

    except Exception as e:
        print(f"Error finding lines in point cloud: {e}")
        return None


def alpha_shape(points, alpha, min_triangle_area=1e-10) -> MultiPoint:
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Uses Delaunay triangulation and circumradius filtering to create a concave boundary
    that can capture inner edges and complex shapes, unlike convex hulls.

    Args:
        points (np.ndarray): Array of shape (n, 2) containing 2D point coordinates.
        alpha (float): Alpha value controlling concavity. Higher values create looser boundaries,
            lower values create tighter, more detailed boundaries.
        min_triangle_area (float, optional): Minimum area threshold to ignore degenerate triangles.
            Defaults to 1e-10.

    Returns:
        shapely.geometry.Polygon or MultiPolygon: The computed alpha shape as a Shapely geometry object.
        Returns convex hull if fewer than 4 input points.

    Example:
        >>> import numpy as np
        >>> # L-shaped building footprint
        >>> points = np.array([[0,0], [2,0], [2,1], [1,1], [1,2], [0,2]])
        >>> boundary = alpha_shape(points, alpha=0.5)
        >>> print(f"Boundary type: {boundary.geom_type}")
        Boundary type: Polygon

    Note:
        - Used internally by find_lines_in_pointcloud for boundary detection
        - Alpha parameter requires tuning: too low = noisy, too high = over-simplified
        - Handles degenerate triangles by skipping those with area below threshold
    """
    if len(points) < 4:
        return MultiPoint(points).convex_hull

    tri = Delaunay(points)
    edges = set()

    with np.errstate(all='ignore'):
        for ia, ib, ic in tri.simplices:
            pa, pb, pc = points[ia], points[ib], points[ic]
            a = np.linalg.norm(pb - pc)
            b = np.linalg.norm(pa - pc)
            c = np.linalg.norm(pa - pb)
            s = (a + b + c) / 2.0
            try:
                area = np.sqrt(s * (s - a) * (s - b) * (s - c))
                if area < min_triangle_area or area <= 0:
                    raise ValueError("Triangle area is too small or zero, skipping this triangle.")
            except Exception:
                # print(f"Error computing area: {e}")
                continue
            circum_r = a * b * c / (4.0 * area)
            if circum_r < 1.0 / alpha:
                # Store edges with sorted indices for uniqueness
                edges.update([tuple(sorted(edge)) for edge in [(ia, ib), (ib, ic), (ic, ia)]])

    edge_lines = [LineString([points[i], points[j]]) for i, j in edges]
    concave = unary_union(polygonize(edge_lines)).buffer(0)
    return concave


def sort_points_in_hull(lines: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """
    Sort points in a point cloud hull based on proximity using nearest neighbor approach.

    Identifies corner points by analyzing distance jumps between consecutive points,
    then reorders them by connecting nearest neighbors to create a coherent spatial sequence.

    Args:
        lines (np.ndarray): Array of shape (N, 3) containing 3D hull/boundary points.
        threshold (float, optional): Distance threshold to consider as a corner. Defaults to 0.1.

    Returns:
        np.ndarray: Array of shape (M, 3) containing sorted corner points,
                   with M <= N. Returns empty array if fewer than 2 input points.

    Raises:
        ValueError: If threshold is less than or equal to 0.

    Example:
        >>> hull_points = np.array([[0.0, 0.0, 0.0], [0.05, 0.02, 0.0], [1.0, 0.0, 0.0]])
        >>> sorted_corners = sort_points_in_hull(hull_points, threshold=0.1)
        >>> print(f"Detected {len(sorted_corners)} corners")

    Note:
        - Uses greedy nearest-neighbor approach which may not be optimal for complex hulls,
          but this is chosen regardless, because of its simplicity and efficiency.
        - Points are reordered starting from leftmost point (maximum x-coordinate)
    """
    # Ensure input is a numpy array
    lines = np.asarray(lines)
    if lines.ndim != 2 or lines.shape[1] < 2:
        raise ValueError("Input to sort_points_in_hull must be a 2D array with at least 2 columns (x, y[, z])")

    if len(lines) < 2:
        return np.array([])

    # Project to 2D (XY)
    points_2d = lines[:, :2]
    n = len(points_2d)
    visited = np.zeros(n, dtype=bool)
    ordered_indices = []

    # Start from the leftmost (min x) point
    current_index = np.argmin(points_2d[:, 0])
    ordered_indices.append(current_index)
    visited[current_index] = True

    for _ in range(1, n):
        current_point = points_2d[current_index]
        # Find the nearest unvisited neighbor in 2D
        dists = np.linalg.norm(points_2d - current_point, axis=1)
        dists[visited] = np.inf
        next_index = np.argmin(dists)
        if dists[next_index] == np.inf:
            break  # No more unvisited points
        ordered_indices.append(next_index)
        visited[next_index] = True
        current_index = next_index

    # Optionally, close the loop if the last point is close to the first
    ordered_points = lines[ordered_indices]
    if np.linalg.norm(ordered_points[0, :2] - ordered_points[-1, :2]) < threshold:
        ordered_points = np.vstack([ordered_points, ordered_points[0]])

    return ordered_points


def create_point_pairs(points: np.ndarray) -> list:
    """
    Creates pairs of points from the input array by connecting each point with the next one.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) containing 3D points.

    Returns:
        list: A list of tuples, where each tuple contains two points (start, end).
    """
    pairs = []
    for i in range(len(points) - 1):
        start = points[i]
        end = points[i + 1]
        pairs.append((start, end))
    return pairs


def get_keypoints(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    salient_radius: float = 0.0,
    non_max_radius: float = 0.0,
    gamma_21: float = 0.975,
    gamma_32: float = 0.975,
    min_neighbors: int = 5,
    print_stats: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    NOTE: This function seems to be non-deterministic which may be due to the nature of the ISS algorithm.
    NOTE: This function has the habbit to crash the entire script from time to time without any message.

    Detects distinctive keypoints in a point cloud using the ISS (Intrinsic Shape Signatures) algorithm.

    This function finds "interest points" in a 3D point cloud—these are points that stand out due to their local geometry,
    such as corners, edges, or other unique features. These keypoints are useful for tasks like matching, registration,
    or simplifying complex point clouds.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): The input point cloud to analyze.
        salient_radius (float): The size of the neighborhood around each point to consider for keypoint detection.
            If set to 0.0, an appropriate value is estimated automatically.
        non_max_radius (float): The radius used to suppress non-maximum responses, ensuring keypoints are well separated.
            If set to 0.0, an appropriate value is estimated automatically.
        gamma_21 (float): Threshold for distinguishing keypoints based on the shape of their local neighborhood.
            Lower values make the detector more selective.
        gamma_32 (float): Similar to gamma_21, controls selectivity based on local geometry.
        min_neighbors (int): Minimum number of neighboring points required for a point to be considered as a keypoint.
        print_stats (bool, optional): If True, prints the number of detected keypoints and other statistics.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: A new point cloud containing only the detected keypoints.

    Raises:
        TypeError: If the input is not an Open3D PointCloud.

    Example:
        >>> keypoints = get_keypoints(pcd, salient_radius=0.1, non_max_radius=0.05)
        >>> print(f"Found {len(keypoints.points)} keypoints")
    """
    if not isinstance(pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud.")

    print("Testing keypoint detection...")

    keypoints = o3d.geometry.keypoint.compute_iss_keypoints(
        pcd,
        salient_radius=salient_radius,
        non_max_radius=non_max_radius,
        gamma_21=gamma_21,
        gamma_32=gamma_32,
        min_neighbors=min_neighbors
    )

    if not keypoints:
        print("No keypoints found. kys")
        return pcd

    if print_stats:
        print(f"Detected {len(keypoints.points)} keypoints from {len(pcd.points)} input points.")
        print(f"Removed {len(pcd.points) - len(keypoints.points)} points.")

    return keypoints


def find_corners(points, angle_threshold_deg=45, window=3, merge_radius=3) -> np.ndarray:
    """
    Detects corner points in a noisy 3D contour by analyzing direction changes and merging nearby detections.

    Args:
        points (np.ndarray): A NumPy array of shape (N, 3) containing 3D points ordered along a contour.
        angle_threshold_deg (float, optional): Minimum angle (in degrees) between smoothed direction vectors
            to consider a corner. Defaults to 45.
        window (int, optional): Number of points to skip forward and backward when computing direction vectors.
            Larger values reduce sensitivity to small noise. Defaults to 3.
        merge_radius (int, optional): Maximum index distance between consecutive corner candidates to consider
            them part of the same corner cluster. Only one point is kept per cluster. Defaults to 3.

    Returns:
        np.ndarray: A NumPy array of shape (M, 3) containing the filtered corner points, including the first
        and last point of the input.

    Example:
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [2, 1, 0], [2, 2, 0]])
        >>> corners = find_corners(points, angle_threshold_deg=45, window=1, merge_radius=1)
        >>> print(corners)

    Note:
        - The function is sensitive to the choice of parameters, especially `angle_threshold_deg` and `window`.
        - Smaller window sizes may detect more corners but can also pick up noise.
    """
    angle_threshold_rad = np.deg2rad(angle_threshold_deg)
    candidate_indices = []

    for i in range(window, len(points) - window):
        vec1 = points[i] - points[i - window]
        vec2 = points[i + window] - points[i]

        vec1 /= np.linalg.norm(vec1)
        vec2 /= np.linalg.norm(vec2)

        angle = np.arccos(np.clip(np.dot(vec1, vec2), -1.0, 1.0))
        if angle > angle_threshold_rad:
            candidate_indices.append(i)

    # Merge nearby candidate indices into a single representative index per corner
    merged_indices = []
    if candidate_indices:
        current_cluster = [candidate_indices[0]]
        for idx in candidate_indices[1:]:
            if idx - current_cluster[-1] <= merge_radius:
                current_cluster.append(idx)
            else:
                merged_indices.append(current_cluster[len(current_cluster) // 2])
                current_cluster = [idx]
        merged_indices.append(current_cluster[len(current_cluster) // 2])  # Final cluster

    # Always include the first and last point
    corner_indices = [0] + merged_indices + [len(points) - 1]
    return points[corner_indices]


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


def contour_to_lineset(points):
    """
    Create a closed LineSet from ordered contour points.
    Args:
        points (np.ndarray): Nx3 array of ordered 3D points.
    Returns:
        o3d.geometry.LineSet
    """
    n = len(points)
    lines = [[i, (i + 1) % n] for i in range(n)]  # closed loop
    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(points)
    lineset.lines = o3d.utility.Vector2iVector(lines)
    return lineset


def connect_vertically_aligned_points(
    floor_points: np.ndarray,
    wall_points: np.ndarray,
    xy_tol: float = 1e-2
) -> o3d.geometry.LineSet:
    """
    Connects points from floor_points to wall_points if their XY coordinates match within a given tolerance.

    Args:
        floor_points (np.ndarray): Nx3 array of floor (lower) points.
        wall_points (np.ndarray): Mx3 array of wall (upper) points.
        xy_tol (float, optional): Tolerance for matching XY coordinates. Defaults to 1e-2.

    Returns:
        o3d.geometry.LineSet: LineSet connecting vertically aligned points.
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
    Connects points from base_level_points to one or more sets of upper_level_points
    if their XY coordinates match within a given tolerance.

    Args:
        base_level_points (np.ndarray): Nx3 array of base (lower) points.
        upper_level_points (Union[np.ndarray, List[np.ndarray]]): Either a single Mx3 array
            or a list of arrays, each representing an upper level.
        xy_tol (float, optional): Tolerance for matching XY coordinates. Defaults to 1e-2.

    Returns:
        o3d.geometry.LineSet: LineSet connecting vertically aligned points.
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


def filter_lines_within_contour(contour_points: np.ndarray, lineset: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    contour_points: Nx3 numpy array that represents the 2D contour in the XY plane.
    lineset: o3d.geometry.LineSet with lines.

    Return: filtered o3d.geometry.LineSet without lines that fall outside the contour.
    """
    # Ensure contour is closed (first point == last point)
    if not np.allclose(contour_points[0], contour_points[-1]):
        contour_points = np.vstack([contour_points, contour_points[0]])

    polygon = Polygon(contour_points[:, :2])

    # Copy points and lines from lineset
    line_points = np.asarray(lineset.points)
    line_indices = np.asarray(lineset.lines)

    new_lines = []
    for (p1_idx, p2_idx) in line_indices:
        p1 = line_points[p1_idx]
        p2 = line_points[p2_idx]

        line = LineString([(p1[0], p1[1]), (p2[0], p2[1])])

        # Check if line is completely within (or on) the polygon
        if polygon.contains(line) or polygon.covers(line):
            new_lines.append([p1_idx, p2_idx])

    # Create new LineSet
    filtered = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(new_lines)
    )

    return filtered


def merge_lineset(*linesets: o3d.geometry.LineSet) -> o3d.geometry.LineSet:
    """
    Merge multiple LineSets into one.
    Args:
        *linesets: Variable number of o3d.geometry.LineSet objects.
    Returns:
        o3d.geometry.LineSet: Combined LineSet.
    """
    all_points = []
    all_lines = []
    all_colors = []
    offset = 0

    for ls in linesets:
        pts = np.asarray(ls.points)
        lines = np.asarray(ls.lines)
        all_points.append(pts)
        all_lines.append(lines + offset)
        if ls.has_colors():
            all_colors.append(np.asarray(ls.colors))
        offset += len(pts)

    merged = o3d.geometry.LineSet()
    merged.points = o3d.utility.Vector3dVector(np.vstack(all_points))
    merged.lines = o3d.utility.Vector2iVector(np.vstack(all_lines))
    if all_colors and all(len(c) == len(p) for c, p in zip(all_colors, all_points)):
        merged.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
    return merged


def export_3d_building_to_cityjson_with_dialog(
    floor_lineset,
    roof_lineset,
    cityjson_properties=None
):
    """Export a 3D building (floor, walls, roof) to CityJSON via a save dialog.

    The floor and roof are exported as polygons. Walls are vertical polygons
    connecting corresponding points on the floor and roof boundaries. The user
    selects the output file via a Tkinter dialog.

    Args:
        floor_lineset (o3d.geometry.LineSet): Open3D LineSet for the floor.
        roof_lineset (o3d.geometry.LineSet): Open3D LineSet for the roof.
        cityjson_properties (dict, optional): Additional CityJSON properties
            (e.g., metadata, attributes). Defaults to None.

    Returns:
        None

    Raises:
        ValueError: If any lineset is empty or not a valid Open3D LineSet.
    """
    def lineset_to_ring(lineset):
        """Convert a LineSet to an ordered closed ring of points."""
        points = np.asarray(lineset.points)
        lines = np.asarray(lineset.lines)
        if len(points) == 0 or len(lines) == 0:
            raise ValueError("LineSet is empty.")
        # Build ordered list of indices from lines
        idx_order = [lines[0][0], lines[0][1]]
        used = set(idx_order)
        for _ in range(len(lines) - 1):
            last = idx_order[-1]
            found = False
            for line in lines:
                if line[0] == last and line[1] not in used:
                    idx_order.append(line[1])
                    used.add(line[1])
                    found = True
                    break
                elif line[1] == last and line[0] not in used:
                    idx_order.append(line[0])
                    used.add(line[0])
                    found = True
                    break
            if not found:
                break
        # Close the ring if not already closed
        if idx_order[0] != idx_order[-1]:
            idx_order.append(idx_order[0])
        return points, idx_order

    # Get ordered rings for floor and roof
    floor_points, floor_ring = lineset_to_ring(floor_lineset)
    roof_points, roof_ring = lineset_to_ring(roof_lineset)

    # Stack all points and deduplicate globally
    all_points_concat = np.vstack([floor_points, roof_points])
    unique_points, inverse = np.unique(
        np.round(all_points_concat, 8), axis=0, return_inverse=True
    )

    # Map local indices to global indices
    floor_offset = 0
    roof_offset = len(floor_points)
    floor_global = [int(inverse[floor_offset + i]) for i in floor_ring]
    roof_global = [int(inverse[roof_offset + i]) for i in roof_ring]

    # Floor and roof polygons (closed rings)
    floor_surface = [floor_global]
    roof_surface = [roof_global]

    # Wall polygons: vertical faces between floor and roof
    wall_surfaces = []
    n = min(len(floor_global), len(roof_global)) - 1  # -1 because last is duplicate (closed ring)
    for i in range(n):
        wall = [
            floor_global[i],
            floor_global[i + 1],
            roof_global[i + 1],
            roof_global[i],
            floor_global[i]
        ]
        wall_surfaces.append([wall])

    # Build CityJSON structure
    cityjson = {
        'type': 'CityJSON',
        'version': '1.1',
        'CityObjects': {
            'building_1': {
                'type': 'Building',
                'geometry': [{
                    'type': 'MultiSurface',
                    'lod': 2,
                    'boundaries': [
                        floor_surface,         # FloorSurface
                        roof_surface,          # RoofSurface
                        *wall_surfaces         # WallSurfaces
                    ],
                    'semantics': {
                        'surfaces': (
                            [{'type': 'FloorSurface'}] +
                            [{'type': 'RoofSurface'}] +
                            [{'type': 'WallSurface'} for _ in wall_surfaces]
                        ),
                        'values': (
                            [[0]] + [[1]] + [[2 + i] for i in range(len(wall_surfaces))]
                        )
                    }
                }],
                'attributes': {
                    'name': 'Example Building'
                }
            }
        },
        'vertices': unique_points.tolist(),
        'metadata': {
            'referenceSystem': 'urn:ogc:def:crs:EPSG::28992'
        }
    }
    if cityjson_properties:
        cityjson.update(cityjson_properties)

    # Tkinter save file dialog
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.asksaveasfilename(
        defaultextension='.json',
        filetypes=[('CityJSON files', '*.json')],
        title='Save CityJSON file'
    )
    if not file_path:
        print("Save cancelled.")
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(cityjson, f, indent=2)
    print(f"CityJSON file saved to {file_path}")


def main():
    # Set the verbosity level of Open3D to only print severe errors
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # 1. Load and preprocess the point cloud (user selects file)
    pcd = load_and_preprocess_pointcloud()

    if pcd is None:
        print("No point cloud loaded. Exiting.")
        return

    # 2. Remove noise from the point cloud
    pcd = rns(pcd)

    # 3. Transform the point cloud into a height map (returns tuple: [floor, wall, ...])
    new_pcd_tuple = transform_pointcloud_to_height_map(
        pcd,
        grid_spacing_cm=2,
        visualize_map=False,
        visualize_map_np=False,
        debugging_logs=False
    )

    # 4. Find the boundary lines (hull) of the floor points
    floor_contour = find_boundary_from_floor(new_pcd_tuple[0], 8)
    print(f"Detected {len(floor_contour)} points in the contour floor point cloud.")

    # 5. Sort the hull points and find corners in the floor boundary
    floor_hull = sort_points_in_hull(floor_contour, 0.045)
    print(f"Detected {len(floor_hull)} points in the floor hull.")

    # Find corners
    floor_corners = find_corners(floor_hull, angle_threshold_deg=45, window=2, merge_radius=1)

    # opce(create_point_cloud(floor_corners, color=[1, 0, 0]), show_help=False)

    temp_merge = merge_pcds(
        [
            get_keypoints(create_point_cloud(floor_contour), gamma_21=0.975, gamma_32=0.975, min_neighbors=3, print_stats=False),
            create_point_cloud(floor_corners, color=[1, 0, 0]),
        ]
    )

    full_floor_corners = sort_points_in_hull(temp_merge.points, 0.00005)

    full_floor_corners = floor_hull  # FOR TESTING

    # 6. Create a wall slice at a certain height above the floor
    floor_corners_pcd = create_point_cloud(full_floor_corners, color=[1, 0, 0])  # Red color for corners
    wall_slice = create_correct_height_slice(
        new_pcd_tuple[1],
        floor_corners_pcd,
        height=1.5,
        search_radius=0.01,
        height_tol=0.75,
        neighbor_window=4,
        min_low_neighbors=6
    )
    wall_floor_merge = merge_pcds([floor_corners_pcd, wall_slice])  # noqa: F841

    # 7. Extract the roof points above a certain height (removes everything below)
    new_roof_pcd = keep_wall_points_from_x_height(
        new_pcd_tuple[1],
        floor_corners_pcd,
        height=1.5
    )

    # 8. Slice the roof into horizontal slabs and flatten each slice
    sliced_roof_list = slice_roof_up(new_roof_pcd, 30, slab_fatness=0.01, voxel_size=0.05)

    # Take the first and second list in sliced_roof_list
    roof_wall_lineset = connect_vertically_aligned_points2(wall_slice.points, sliced_roof_list, 0.1)

    # Connect the rest of the roof layers with each other from top to the bottom and per layer create a contour
    for i in range(len(sliced_roof_list) - 1, 0, -1):
        roof_wall_lineset += connect_vertically_aligned_points(sliced_roof_list[i - 1], sliced_roof_list[i], 0.1)
        roof_wall_lineset += contour_to_lineset(sort_points_in_hull(sliced_roof_list[i]))

    roof_wall_lineset = filter_lines_within_contour(full_floor_corners, roof_wall_lineset)

    floor_lineset = contour_to_lineset(full_floor_corners)
    wall_slice_lineset = contour_to_lineset(sort_points_in_hull(wall_slice.points, 0.00005))
    vertical_lineset = connect_vertically_aligned_points(floor_lineset.points, wall_slice_lineset.points)

    # Combine vertical and wall slice lineset
    combined_lineset = merge_lineset(vertical_lineset, wall_slice_lineset)  # noqa: F841

    # o3d.visualization.draw([
    #     floor_lineset,
    #     combined_lineset,
    #     roof_wall_lineset
    # ])

    export_3d_building_to_cityjson_with_dialog(floor_lineset, combined_lineset, roof_wall_lineset)


if __name__ == "__main__":
    main()
