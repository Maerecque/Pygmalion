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
    Loads a pointcloud file selected by the user and applies voxel downsampling and normal estimation.
    Returns:
        o3d.geometry.PointCloud: The processed point cloud.
    """
    pcd = readout_LAS_file(get_file_path("Select a point cloud file to process", "LAS files (*.las *.laz)"))
    print("Point cloud loaded and preprocessed. Now starting surface reconstruction...")
    return pcd


def alpha_shape(points, alpha, min_triangle_area=1e-10):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.
    This function is used in find_lines_in_pointcloud to detect boundaries in a 2D projection of a 3D point cloud.

    Args:
        points: np.array of shape (n, 2)
        alpha: alpha value controlling concavity (higher = looser)
        min_triangle_area: threshold to ignore near-degenerate triangles

    Returns:
        A Shapely Polygon or MultiPolygon representing the alpha shape.
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
    """Find lines in a 3D point cloud by projecting to 2D and detecting boundaries.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        alpha (float, optional): The alpha parameter for the alpha shape. Defaults to 10.
        min_triangle_area (float, optional): Minimum area of triangles to consider. Defaults to 1e-10.

    Raises:
        TypeError: If the input is not a valid Open3D PointCloud.
        ValueError: If the point cloud is empty.
        TypeError: If the point cloud is not of type Vector3dVector.

    Returns:
        np.ndarray: The detected line points.
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
    Sort points in a point cloud hull based on their proximity to each other.

    Args:
        lines (np.ndarray): The detected line points.
        threshold (float, optional): Distance threshold to consider as a corner. Defaults to 0.1.

    Returns:
        np.ndarray: An array of sorted points based on proximity.
    """
    # If there are fewer than 2 points, no order can be found
    if len(lines) < 2:
        return np.array([])

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


def get_extent(points: np.ndarray) -> dict:
    """Get the spatial extent of a set of 3D points.

    Meant for CityJSON metadata generation.

    Args:
        points (np.ndarray): The input point cloud data.

    Returns:
        dict: A dictionary containing the minimum and maximum coordinates of the point cloud.
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


def find_corners_clean(points, angle_threshold_deg=45, window=3, merge_radius=3):
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
        >>> corners = find_corners_clean(points, angle_threshold_deg=45, window=1, merge_radius=1)
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
    """Create pairs of points from a 2D array of points with the index of each point.

    Args:
        points (np.ndarray): A 2D array of shape (N, 3) containing 3D points.

    Returns:
        np.ndarray: A 2D array of shape (M, 2, 3) where M is the number of pairs.

    Raises:
        TypeError: If the input is not a NumPy array.
        ValueError: If the input array is empty or does not have the correct shape.
        ValueError: If there are fewer than two points to create pairs.
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
    if generalize:
        # Floor
        flr_hull = sort_points_in_hull(points, 0.05)
        flr_corners = find_corners_clean(flr_hull, angle_threshold_deg=45, window=2, merge_radius=1)
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

    Args:
        tbp_pcd (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be sliced.
        floor_contour_pcd (o3d.cpu.pybind.geometry.PointCloud): The floor point cloud to reference for height.
        height (float): The height at which to slice the point cloud in meters. Defaults to 1.5.
        search_radius (float): The radius within which to search for points in the tbp_pcd.
            Must be a positive float. Defaults to 0.025.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: A new point cloud containing the sliced points.
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
    """Keep wall points above a certain height.

    Args:
        wall_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing wall points.
        floor_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing floor points.
        height (float, optional): Height above the floor to keep wall points. Defaults to 1.5.

    Raises:
        TypeError: If the input is not an Open3D PointCloud object.
        ValueError: If the input point cloud is empty.
        ValueError: If height is not positive.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud containing wall points above the specified height.
    """
    # Step one, get height of the floor from the floor point cloud
    # Step two, add given height to the floor height (height + floor_height) = minimum z value of the ceiling points
    # Step three, filter the wall point cloud to keep only points above the minimum z value
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
    Find the highest point in `full_pcd` that lies approximately above each point in `corner_pcd`.

    For each corner point, this function looks for points in `full_pcd` within a horizontal
    distance ±`search_radius` in both x and y directions and selects the point with the highest z-coordinate.

    Args:
        corner_pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud containing corner points.
        full_pcd (o3d.cpu.pybind.geometry.PointCloud): Larger point cloud to search for points above corners.
        search_radius (float, optional): Search radius for finding points above corners. Must be greater than 0.
            Defaults to 0.01.
        compare_with_corner (bool, optional): If True, compare the resulting highest points with the corner points.
            The user should define what happens with this boolean.

    Raises:
        TypeError: If inputs are not Open3D PointCloud objects.
        ValueError: If either point cloud is empty.
        ValueError: If no points are found above any corner point.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: A point cloud containing the highest points above each corner point,
            colored red for visualization.
    """

    # Validate input types
    if not isinstance(corner_pcd, o3d.cpu.pybind.geometry.PointCloud) or not isinstance(full_pcd, o3d.cpu.pybind.geometry.PointCloud):
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

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): Input point cloud containing ceiling points.
        percentage_to_keep (float, optional): Fraction of points to keep (0 < percentage_to_keep <= 1). Defaults to 0.1.

    Raises:
        TypeError: If input_pcd is not an Open3D PointCloud.
        ValueError: If input_pcd is empty.
        ValueError: If percentage_to_keep is not in (0, 1].

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Filtered point cloud with the top ceiling points.
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
    If `search_radius` is provided, choose the highest point among all ceiling points within that radius (XY),
    otherwise fall back to the single nearest neighbor (in XY).

    Args:
        ceiling_pcd: Open3D point cloud with ceiling points (Nx3).
        floor_corners: (M,3) numpy array of floor corner coordinates.
        search_radius: optional float. If set, consider only points within this radius (XY-plane). Default is 0.3

    Returns:
        Open3D point cloud containing one point (highest) per corner-pair.
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
    Slice a point cloud along Z into horizontal slices, then run find_lines_in_pointcloud
    on each slice to detect line points and combine all detected line points into one point cloud.

    Args:
        roof_pcd (open3d.cpu.pybind.geometry.PointCloud): Input point cloud.
        slices_amount (int): Number of horizontal slices.
        slab_fatness (float): Half-fatness around slice center to include points (default 0.01).

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: New point cloud containing all detected line points.
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
    floor_corners = find_corners_clean(floor_hull, angle_threshold_deg=45, window=2, merge_radius=1)

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

    ### FLOOR EXPORT SECTION START ###
    floor_corners_pcd_array = np.asarray(floor_corners_pcd.points)
    wall_points_array = np.asarray(wall_hull_pcd.points)
    highest_ridge_points_array = np.asarray(highest_ridge_points.points)
    floor_lineset = create_lineset_from_contour(floor_corners_pcd_array, False)
    # opce(floor_lineset)  # Display the floor lineset
    wall_lineset = create_lineset_from_contour(wall_points_array, False)
    # opce(wall_lineset)  # Display the wall lineset
    highest_ridge_points_lineset = create_lineset_from_contour(highest_ridge_points_array, False)
    # opce(highest_ridge_points_lineset)  # Display the highest ridge points lineset

    # opce([floor_lineset, wall_lineset, highest_ridge_points_lineset])  # Display the linesets

    # # Show how many points are in the lineset
    # print(f"Lineset contains {len(floor_lineset.points)} points and {len(floor_lineset.lines)} lines.")

    # opce([floor_lineset])  # Display the lineset and hull

    ### FLOOR EXPORT SECTION END ###

    # new_small_ceiling_pcd = filter_ceiling_points(new_ceiling_pcd, percentage_to_keep=0.02)

    all_merge = merge_pcds([wall_floor_merge, highest_ridge_points])
    opce(all_merge)

    # get_extent(floor_hull)

    # floor_corners = floor_corners[:(len(floor_corners) // 2)]

    # floor_corners_pcd = create_point_cloud(floor_corners)
    # opce(floor_corners_pcd)

    # wall_pcd = new_pcd_tuple[1]
    # ceiling_pcd = new_pcd_tuple[2]


if __name__ == "__main__":
    main()
