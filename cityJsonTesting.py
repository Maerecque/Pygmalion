import open3d as o3d
import sys
import os
import numpy as np  # noqa: F401
from tqdm import tqdm

# Use alpha shape to find concave boundary (captures inner/underlying edges)
from shapely.geometry import MultiPoint, LineString
from shapely.ops import unary_union, polygonize
from scipy.spatial import Delaunay
from scipy.spatial import cKDTree


sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

from Source.fileHandler import get_file_path, readout_LAS_file
from Source.heightMapModule import transform_pointcloud_to_height_map, create_point_cloud
from Source.pointCloudEditor import open_point_cloud_editor as opce
from Source.pointCloudAltering import remove_noise_statistical as rns, merge_point_clouds as merge_pcds  # , grid_subsampling


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


def find_lines_in_pointcloud(pcd: o3d.geometry.PointCloud, alpha: float = 10, min_triangle_area: float = 1e-10) -> np.ndarray:
    """
    Find boundary/contour lines in a 3D point cloud by projecting to 2D and detecting alpha shape boundaries.

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
        - Uses greedy nearest-neighbor approach which may not be optimal for complex hulls
        - Points are reordered starting from leftmost point (maximum x-coordinate)
    """
    # If there are fewer than 2 points, no order can be found
    if len(lines) < 2:
        return np.array([])

    if threshold <= 0:
        raise ValueError("Threshold must be greater than 0.")

    pnts = []
    # Iterate through the line points and check the distance between consecutive points
    for i in range(1, len(lines)):
        dist = np.linalg.norm(lines[i] - lines[i - 1])
        # If the distance exceeds the threshold, consider it a corner
        if dist > threshold:
            pnts.append(lines[i])

    # If no points were found, return an empty array
    if not pnts:
        return np.array([])

    pnts = np.array(pnts)
    # Build a KD-tree for efficient nearest neighbor search
    tree = cKDTree(pnts)

    ordered_pnts = []
    visited = np.zeros(len(pnts), dtype=bool)
    current_index = 0
    # Start ordering from the first point
    ordered_pnts.append(pnts[current_index])
    visited[current_index] = True

    # Greedily order points by always picking the nearest unvisited neighbor
    for _ in range(1, len(pnts)):
        distances, indices = tree.query(pnts[current_index], k=len(pnts))
        for idx in indices:
            if not visited[idx]:
                ordered_pnts.append(pnts[idx])
                visited[idx] = True
                current_index = idx
                break

    # Ensure the points are ordered in a consistent manner, by sorting by x-coordinate.
    # This assumes the first point is the leftmost point. Works well for most cases.
    max_x_index = np.argmax([pnt[0] for pnt in ordered_pnts])
    ordered_pnts = np.roll(ordered_pnts, -max_x_index, axis=0)

    return np.array(ordered_pnts)


def get_extent(points: np.ndarray) -> np.ndarray:
    """
    Get the spatial extent (bounding box) of 3D points in CityJSON format.

    Computes min/max coordinates and returns them in CityJSON metadata order:
    [min_x, min_z, min_y, max_x, max_z, max_y]. Note the Y/Z coordinate swap.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing 3D point coordinates [x, y, z].

    Returns:
        np.ndarray: Array of shape (6,) containing spatial extent in CityJSON format.

    Raises:
        TypeError: If input is not a numpy array.
        ValueError: If the input array is empty or doesn't have shape (N, 3).

    Example:
        >>> building_points = np.array([[10.0, 20.0, 0.0], [15.0, 25.0, 3.0]])
        >>> extent = get_extent(building_points)
        >>> print("CityJSON extent:", extent)
        CityJSON extent: [10.  0. 20. 15.  3. 25.]

    Note:
        - CityJSON format swaps Y and Z coordinates: [min_x, min_z, min_y, max_x, max_z, max_y]
        - Standard 3D extent would be [min_x, min_y, min_z, max_x, max_y, max_z]
    """
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_coords = points[:, 2]
    extends = np.array([
        np.min(x_coords),
        np.min(z_coords),
        np.min(y_coords),
        np.max(x_coords),
        np.max(z_coords),
        np.max(y_coords)
    ])
    return extends


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


def create_point_pairs(points: np.ndarray) -> np.ndarray:
    """
    Create pairs of consecutive point indices for connecting points in a contour.

    Creates index pairs for each consecutive point (0->1, 1->2, etc.) and closes
    the loop by connecting the last point back to the first point.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing 3D points.

    Returns:
        np.ndarray: Array of shape (N, 2) containing index pairs for connecting points.
                   Each row contains [current_index, next_index] for creating line segments.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the input array is empty or doesn't have shape (N, 3).
        ValueError: If there are fewer than two points to create pairs.

    Example:
        >>> points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> pairs = create_point_pairs(points)
        >>> print(pairs)
        [[0 1]
         [1 2]
         [2 3]
         [3 0]]  # Closes the loop

    Note:
        - Automatically closes the contour by connecting last point to first
        - Returns index pairs, not the actual point coordinates
        - Useful for creating Open3D LineSet objects from contour points
    """
    try:
        if not isinstance(points, np.ndarray):
            raise TypeError("Input must be a NumPy array.")
        if points.size == 0:
            raise ValueError("Input array is empty.")
        if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("Input must be a 2D NumPy array with shape (N, 3).")
        if len(points) < 2:
            raise ValueError("At least two points are required to create pairs.")
    except Exception as e:
        print(f"Error in create_point_pairs: {e}")
        return np.array([])

    pairs = []
    for i in range(len(points) - 1):
        pairs.append([i, i + 1])

    # Close the loop
    pairs.append([len(points) - 1, 0])
    return np.array(pairs)


def create_lineset_from_contour(points: np.ndarray, generalize=True) -> o3d.geometry.LineSet:
    """
    Create an Open3D LineSet from contour points with optional generalization.

    Processes input points to create a connected line visualization. When generalization
    is enabled, sorts points by proximity and detects corners to simplify the contour.

    Args:
        points (np.ndarray): Array of shape (N, 3) containing 3D contour points.
        generalize (bool, optional): Whether to apply hull sorting and corner detection
            to simplify the contour. Defaults to True.

    Returns:
        o3d.geometry.LineSet: LineSet object with points and line connections for visualization.

    Example:
        >>> contour_points = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
        >>> lineset = create_lineset_from_contour(contour_points, generalize=True)
        >>> print(f"Created lineset with {len(lineset.points)} points")
        Created lineset with 4 points

    Example without generalization:
        >>> raw_lineset = create_lineset_from_contour(contour_points, generalize=False)
        >>> # Uses points as-is without hull sorting or corner detection

    Note:
        - With generalization: applies sort_points_in_hull() and find_corners() for cleaner contours
        - Without generalization: uses input points directly with create_point_pairs()
        - Prints detected corner points and connection pairs for debugging
        - Automatically creates closed contours by connecting last point to first
    """
    if generalize:
        # Floor
        flr_hull = sort_points_in_hull(points, 0.05)
        flr_corners = find_corners(flr_hull, angle_threshold_deg=45, window=2, merge_radius=1)
        points = flr_corners

    # Show the points in the corners in print statement, print the points without e+ notation
    flr_corners = np.array([np.round(corner, 3) for corner in points])
    print(f"Floor corners: {flr_corners.tolist()}")

    # Not sure about this line, it's a bit too early. The previous two lines work perfect for the floor.
    flr_pnt_pairs = create_point_pairs(flr_corners)

    print(f"Creating lineset from {len(flr_corners)} corner points and {len(flr_pnt_pairs)} pairs.")

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(flr_corners)
    lines.lines = o3d.utility.Vector2iVector(flr_pnt_pairs)
    return lines


def create_correct_height_slice(
    tbp_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_contour_pcd: o3d.cpu.pybind.geometry.PointCloud,
    height: float = 1.5,
    search_radius: float = 0.025
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
        - Returns blue-colored point cloud for visualization
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
            print(f"Found {len(candidates)} candidates for floor point {floor_pt}.")
            # Pick the closest point that is not above the slice height but as close as possible
            below_slice = candidates[candidates[:, 2] <= slice_height]
            if below_slice.size > 0:
                print(f"Found {len(below_slice)} candidates below the slice height for floor point {floor_pt}.")

                idx = np.argmin(np.abs(below_slice[:, 2] - slice_height))

                # If there are more than one point near the floor point, pick the one closest to the floor point
                if below_slice.shape[0] > 1:
                    distances = np.linalg.norm(below_slice[:, :2] - floor_pt[:2], axis=1)
                    idx = np.argmin(distances)

                tbp_countour_points.append(below_slice[idx])
            else:
                # If there are no points at or below the slice height the ceiling should be above the slice height
                # Which would mean you can place a point at the slice height
                new_point = np.array([floor_pt[0], floor_pt[1], slice_height])
                tbp_countour_points.append(new_point)

    tbp_countour_points = np.array(tbp_countour_points)

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
        - Returns red-colored points for visualization
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


def filter_ceiling_points(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    percentage_to_keep: float = 0.1
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Filter ceiling points by retaining only the top percentage of points based on their z-coordinates.

    Sorts all points by height and keeps only the specified percentage of highest points,
    effectively filtering out lower structural elements and retaining ceiling points.

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): Input point cloud containing ceiling points.
        percentage_to_keep (float, optional): Fraction of points to keep (0 < percentage_to_keep <= 1).
            Defaults to 0.1.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Filtered point cloud with the top ceiling points,
                                           colored blue for visualization.

    Raises:
        TypeError: If input_pcd is not an Open3D PointCloud.
        ValueError: If input_pcd is empty.
        ValueError: If percentage_to_keep is not in (0, 1].

    Example:
        >>> ceiling_pcd = create_point_cloud(ceiling_points, color=[0, 1, 0])
        >>> filtered_ceiling = filter_ceiling_points(ceiling_pcd, percentage_to_keep=0.05)
        >>> print(f"Kept {len(filtered_ceiling.points)} ceiling points (top 5%)")
        Kept 156 ceiling points (top 5%)

    Note:
        - Sorts points by Z-coordinate and selects the highest percentage
        - All filtered points are colored blue for visualization
        - Useful for removing noise and focusing on actual ceiling structure
        - Higher percentages keep more points but may include non-ceiling elements
    """

    # Validate input type and parameters
    if not isinstance(input_pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object.")
    if len(input_pcd.points) == 0:
        raise ValueError("Input point cloud is empty; cannot filter ceiling points.")
    if not (0 < percentage_to_keep <= 1):
        raise ValueError("percentage_to_keep must be between 0 (exclusive) and 1 (inclusive).")

    # Convert points to numpy array for processing
    points_np = np.asarray(input_pcd.points)

    # Sort indices by z-coordinate in ascending order
    sorted_indices = np.argsort(points_np[:, 2])

    # Determine how many points to keep from the top (highest z)
    num_to_keep = int(len(points_np) * percentage_to_keep)

    # Select indices corresponding to the highest z values
    top_indices = sorted_indices[-num_to_keep:]

    # Extract the filtered point cloud
    filtered_pcd = input_pcd.select_by_index(top_indices)

    # Color filtered points blue for visualization
    blue_color = np.array([0, 0, 1])
    colors = np.tile(blue_color, (len(filtered_pcd.points), 1))
    filtered_pcd.colors = o3d.utility.Vector3dVector(colors)

    return filtered_pcd


def find_highest_point_above_point_pair(
    ceiling_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_corners: np.ndarray,
    search_radius: float = 0.3
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    For each pair of floor corner indices (i,i+1) compute the midpoint and find the highest ceiling point above that midpoint.

    Creates consecutive pairs of floor corners, computes their midpoints, and searches for the highest
    ceiling point within the search radius (XY plane) above each midpoint.
    Falls back to nearest neighbor if no points found within radius.

    Args:
        ceiling_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing ceiling points to search.
        floor_corners (np.ndarray): Array of shape (M, 3) containing floor corner coordinates.
        search_radius (float, optional): Search radius in XY plane for finding points above midpoints.
            If None, uses nearest neighbor. Defaults to 0.3.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud containing one highest point per corner pair.

    Raises:
        TypeError: If ceiling_pcd is not an Open3D PointCloud or floor_corners is not a NumPy array.
        ValueError: If ceiling_pcd is empty, floor_corners doesn't have shape (M, 3), or fewer than 2 corners provided.

    Example:
        >>> floor_corners = np.array([[0,0,0], [1,0,0], [1,1,0], [0,1,0]])
        >>> ceiling_pcd = create_point_cloud(ceiling_points, color=[0,1,0])
        >>> ridge_points = find_highest_point_above_point_pair(ceiling_pcd, floor_corners, search_radius=0.2)
        >>> print(f"Found {len(ridge_points.points)} ridge points")
        Found 4 ridge points

    Note:
        - Creates pairs using create_point_pairs() which closes the loop (last->first)
        - Searches in XY plane only, selects highest Z-coordinate from candidates
        - Falls back to nearest neighbor if no points within search_radius
        - Useful for finding roof ridge points above building edge midpoints
    """
    # basic validation
    if not isinstance(ceiling_pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("ceiling_pcd must be an Open3D PointCloud.")
    if len(ceiling_pcd.points) == 0:
        raise ValueError("ceiling_pcd is empty.")
    if not isinstance(floor_corners, np.ndarray):
        raise TypeError("floor_corners must be a NumPy array.")
    if floor_corners.ndim != 2 or floor_corners.shape[1] != 3:
        raise ValueError("floor_corners must have shape (M, 3).")
    if len(floor_corners) < 2:
        raise ValueError("At least two floor corners required to form pairs.")

    corner_pairs = create_point_pairs(floor_corners)
    ceiling_points = np.asarray(ceiling_pcd.points)  # (N,3)
    highest_points = []

    # Optionally you can create an Open3D KDTree for faster spatial queries on big clouds:
    # kdtree = o3d.geometry.KDTreeFlann(ceiling_pcd)

    for pair in corner_pairs:
        i0, i1 = int(pair[0]), int(pair[1])  # indices into floor_corners
        p0 = floor_corners[i0]
        p1 = floor_corners[i1]
        middle_point = (p0 + p1) / 2.0  # (3,)

        # XY distances from ceiling points to the midpoint
        vec_xy = ceiling_points[:, :2] - middle_point[:2]
        dists_xy = np.linalg.norm(vec_xy, axis=1)

        if search_radius is None:
            # pick nearest neighbor in XY
            closest_idx = int(np.argmin(dists_xy))
            highest_points.append(ceiling_points[closest_idx])
        else:
            # consider only points within search_radius (XY)
            mask = dists_xy <= float(search_radius)
            if np.any(mask):
                candidates = ceiling_points[mask]
                # pick the candidate with maximum Z (highest)
                max_z_idx = int(np.argmax(candidates[:, 2]))
                highest_points.append(candidates[max_z_idx])
            else:
                # fallback to nearest if nothing in radius
                closest_idx = int(np.argmin(dists_xy))
                highest_points.append(ceiling_points[closest_idx])

    highest_points = np.array(highest_points, dtype=float)
    highest_points_pcd = o3d.cpu.pybind.geometry.PointCloud()
    highest_points_pcd.points = o3d.utility.Vector3dVector(highest_points)
    return highest_points_pcd


def slice_roof_up(
    roof_pcd: o3d.cpu.pybind.geometry.PointCloud,
    slices_amount: int = 2,
    slab_fatness: float = 0.01
) -> o3d.cpu.pybind.geometry.PointCloud:
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

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: New point cloud containing all flattened slice points.
                                           Returns empty point cloud if no points found in any slice.

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
        slice_points[:, 2] = z_center

        all_flattened_points.append(slice_points)

    if len(all_flattened_points) == 0:
        # No points found in any slice, return empty PointCloud
        return o3d.geometry.PointCloud()

    combined_points = np.vstack(all_flattened_points)

    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(combined_points)

    return new_pcd


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
        grid_size=500,
        visualize_map=False,
        debugging_logs=False
    )

    # 4. Find the boundary lines (hull) of the floor points
    floor_lines = find_lines_in_pointcloud(new_pcd_tuple[0], 11)
    print(f"Detected {len(floor_lines)} lines in the floor point cloud.")  # Lies

    # 5. Sort the hull points and find corners in the floor boundary
    floor_hull = sort_points_in_hull(floor_lines, 0.05)
    floor_corners = find_corners(floor_hull, angle_threshold_deg=45, window=2, merge_radius=1)

    print(f"Detected {len(floor_hull)} points in the floor hull.")
    print(f"Detected {len(floor_corners)} corners in the floor hull.")

    # 6. Create a wall slice at a certain height above the floor
    wall_slice = create_correct_height_slice(new_pcd_tuple[1], create_point_cloud(floor_corners, color=[1, 0, 0]), height=1.5)
    floor_corners_pcd = create_point_cloud(floor_corners, color=[1, 0, 0])  # Red color for corners
    wall_floor_merge = merge_pcds([floor_corners_pcd, wall_slice])

    # 7. Extract the roof points above a certain height (removes everything below)
    new_roof_pcd = keep_wall_points_from_x_height(
        new_pcd_tuple[1],
        floor_corners_pcd,
        height=1.5
    )

    # 8. Slice the roof into horizontal slabs and flatten each slice
    sliced_roof = slice_roof_up(new_roof_pcd, 5, slab_fatness=0.0075)

    # 9. For each hull point, keep the highest point in the sliced roof (find roof outline)
    filtered_sliced_roof = keep_highest_point_above_corner(create_point_cloud(floor_hull), sliced_roof, 0.025, True)

    # 10. Merge the wall, floor, and roof outline for visualization
    combine_till_here = merge_pcds([wall_floor_merge, filtered_sliced_roof])
    opce(combine_till_here)

    exit()

    # ### FLOOR EXPORT SECTION START ###
    # floor_corners_pcd_array = np.asarray(floor_corners_pcd.points)
    # wall_points_array = np.asarray(wall_slice.points)
    # highest_ridge_points_array = np.asarray(highest_ridge_points.points)
    # floor_lineset = create_lineset_from_contour(floor_corners_pcd_array, False)
    # # opce(floor_lineset)  # Display the floor lineset
    # wall_lineset = create_lineset_from_contour(wall_points_array, False)
    # # opce(wall_lineset)  # Display the wall lineset
    # highest_ridge_points_lineset = create_lineset_from_contour(highest_ridge_points_array, False)
    # # opce(highest_ridge_points_lineset)  # Display the highest ridge points lineset

    # # opce([floor_lineset, wall_lineset, highest_ridge_points_lineset])  # Display the linesets

    # # # Show how many points are in the lineset
    # # print(f"Lineset contains {len(floor_lineset.points)} points and {len(floor_lineset.lines)} lines.")

    # # opce([floor_lineset])  # Display the lineset and hull

    # ### FLOOR EXPORT SECTION END ###

    # # new_small_ceiling_pcd = filter_ceiling_points(new_ceiling_pcd, percentage_to_keep=0.02)

    # all_merge = merge_pcds([wall_floor_merge, highest_ridge_points])
    # opce(all_merge)

    # # get_extent(floor_hull)

    # # floor_corners = floor_corners[:(len(floor_corners) // 2)]

    # # floor_corners_pcd = create_point_cloud(floor_corners)
    # # opce(floor_corners_pcd)

    # # wall_pcd = new_pcd_tuple[1]
    # # ceiling_pcd = new_pcd_tuple[2]


if __name__ == "__main__":
    main()
