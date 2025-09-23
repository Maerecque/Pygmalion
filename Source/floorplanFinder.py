"""
floorplanFinder.py

Boundary and keypoint detection utilities for 3D floorplan point clouds using Open3D, Shapely, and SciPy.

Modules:
    - numpy
    - open3d
    - scipy.spatial
    - shapely.geometry
    - shapely.ops

Imports:
    - Delaunay, cKDTree: Used for triangulation and nearest neighbor search in boundary detection.
    - unary_union, polygonize: Used for constructing alpha shapes (concave hulls) from edge sets.
    - MultiPoint, LineString: Used for geometric operations and hull construction.

Functions:
    - find_boundary_from_floor: Projects a 3D point cloud to 2D, detects boundary points using alpha shapes, and returns 3D
      boundary coordinates.
    - alpha_shape: Computes the alpha shape (concave hull) of a set of 2D points using Delaunay triangulation and circumradius
      filtering.
    - sort_points_in_hull: Sorts boundary points in spatial order using a greedy nearest-neighbor approach, optionally closing
      the loop.
    - get_keypoints: Detects distinctive keypoints in a point cloud using the ISS (Intrinsic Shape Signatures) algorithm.
    - find_corners: Detects corner points in an ordered 3D contour by analyzing direction changes and merging nearby detections.

Typical Usage:
    1. Use find_boundary_from_floor to extract boundary points from a 3D floorplan point cloud.
    2. Use sort_points_in_hull to spatially order boundary points for further processing.
    3. Use get_keypoints to detect salient keypoints in the point cloud for matching or registration.
    4. Use find_corners to identify corner points in noisy contours for geometric analysis or simplification.

See individual function docstrings for details.
"""

import numpy as np
import open3d as o3d
from scipy.spatial import Delaunay, cKDTree
from shapely.ops import unary_union, polygonize
from shapely.geometry import MultiPoint, LineString


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
    NOTE: I think I have fixed the crashing by using a newer version of open3d.

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
