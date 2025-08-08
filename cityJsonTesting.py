import open3d as o3d
import sys
import os
import numpy as np  # noqa: F401

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


def create_lineset_from_contour(points: np.ndarray) -> o3d.geometry.LineSet:
    # Floor
    flr_hull = sort_points_in_hull(points, 0.05)
    flr_corners = find_corners_clean(flr_hull, angle_threshold_deg=45, window=2, merge_radius=1)

    # Show the points in the corners in print statement, print the points without e+ notation
    flr_corners = np.array([np.round(corner, 3) for corner in flr_corners])
    print(f"Floor corners: {flr_corners.tolist()}")

    # Not sure about this line, it's a bit too early. The previous two lines work perfect for the floor.
    flr_pnt_pairs = create_point_pairs(flr_corners)

    print(f"Creating lineset from {len(flr_corners)} corner points and {len(flr_pnt_pairs)} pairs.")

    lines = o3d.geometry.LineSet()
    lines.points = o3d.utility.Vector3dVector(flr_corners)
    lines.lines = o3d.utility.Vector2iVector(flr_pnt_pairs)
    return lines


def create_correct_height_wall_slice(points: np.ndarray, height: float = 1.5) -> np.ndarray:
    """
    Create a slice of the point cloud at a specified height above the floor.
    Args:
        points (np.ndarray): A 2D array of shape (N, 3) containing 3D points.
        height (float): The height at which to slice the point cloud. Defaults to 1.5.

    Returns:
        np.ndarray: A 2D array of shape (M, 3) containing the sliced points at the specified height.
    """
    # So this places an identical set of points on top of the floor points,
    # if the points are originally are at 0,8m and the height is 1.5m,
    # the points will be at 0.8 + 1.5 = 2.3
    if not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Input must be a 2D NumPy array with shape (N, 3).")

    if len(points) == 0:
        raise ValueError("Input array is empty. Cannot create a wall slice.")

    # Create a new array with the same shape as points
    wall_slice = np.copy(points)

    # Add the height to the z-coordinate of each point
    wall_slice[:, 2] += height

    return wall_slice


def keep_ceiling_points_from_x_height(
    ceiling_pcd: o3d.cpu.pybind.geometry.PointCloud,
    floor_pcd: o3d.cpu.pybind.geometry.PointCloud,
    height: float = 1.5
) -> o3d.cpu.pybind.geometry.PointCloud:
    # Step one, get height of the floor from the floor point cloud
    # Step two, add given height to the floor height (height + floor_height) = minimum z value of the ceiling points
    # Step three, filter the ceiling point cloud to keep only points above the minimum z value
    if not isinstance(
        ceiling_pcd, o3d.cpu.pybind.geometry.PointCloud
    ) or not isinstance(
        floor_pcd, o3d.cpu.pybind.geometry.PointCloud
    ):
        raise TypeError("Both ceiling_pcd and floor_pcd must be Open3D PointCloud objects.")
    if len(ceiling_pcd.points) == 0 or len(floor_pcd.points) == 0:
        raise ValueError("Both ceiling_pcd and floor_pcd must contain points.")
    # Get the minimum z value from the floor point cloud
    floor_height = np.min(np.asarray(floor_pcd.points)[:, 2])

    # Calculate the minimum z value for the ceiling points
    min_z_value = floor_height + height

    # Filter the ceiling point cloud to keep only points above the minimum z value
    ceiling_points = np.asarray(ceiling_pcd.points)
    ceiling_points_filtered = ceiling_points[ceiling_points[:, 2] > min_z_value]

    if ceiling_points_filtered.size == 0:
        raise ValueError("No ceiling points found above the specified height.")

    # Create a new point cloud with the filtered points
    filtered_ceiling_pcd = o3d.cpu.pybind.geometry.PointCloud()
    filtered_ceiling_pcd.points = o3d.utility.Vector3dVector(ceiling_points_filtered)
    # Make all new points blue
    filtered_ceiling_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(filtered_ceiling_pcd.points), 1)))

    return filtered_ceiling_pcd


def filter_ceiling_points(
    input_pcd: o3d.cpu.pybind.geometry.PointCloud,
    percentage_to_keep: float = 0.1
) -> o3d.cpu.pybind.geometry.PointCloud:
    # Sort the points by their z-coordinate
    # Keep only a percentage of the points with the highest z-coordinates
    if not isinstance(input_pcd, o3d.cpu.pybind.geometry.PointCloud):
        raise TypeError("Input must be an Open3D PointCloud object.")
    if len(input_pcd.points) == 0:
        raise ValueError("Input point cloud is empty. Cannot filter ceiling points.")
    if not (0 < percentage_to_keep <= 1):
        raise ValueError("percentage_to_keep must be between 0 and 1.")
    points = np.asarray(input_pcd.points)
    # Sort points by z-coordinate
    sorted_indices = np.argsort(points[:, 2])
    # Calculate the number of points to keep
    num_points_to_keep = int(len(points) * percentage_to_keep)
    # Get the indices of the points to keep
    indices_to_keep = sorted_indices[-num_points_to_keep:]
    # Select the points to keep
    filtered_pcd = input_pcd.select_by_index(indices_to_keep)
    # Make all new points blue
    filtered_pcd.colors = o3d.utility.Vector3dVector(np.tile([0, 0, 1], (len(filtered_pcd.points), 1)))
    return filtered_pcd


def main():
    pcd = load_and_preprocess_pointcloud()

    if pcd is None:
        print("No point cloud loaded. Exiting.")
        return

    # pcd = grid_subsampling(pcd, 0.1)
    pcd = rns(pcd)

    new_pcd_tuple = transform_pointcloud_to_height_map(
        pcd,
        grid_size=500,
        visualize_map=False,
        debugging_logs=False
    )

    new_pcd = merge_pcds(new_pcd_tuple)
    # opce(new_pcd)

    floor_lines = find_lines_in_pointcloud(new_pcd_tuple[0], 11)
    print(f"Detected {len(floor_lines)} lines in the floor point cloud.")  # Lies

    floor_hull = sort_points_in_hull(floor_lines, 0.05)
    floor_corners = find_corners_clean(floor_hull, angle_threshold_deg=45, window=2, merge_radius=1)

    print(f"Detected {len(floor_hull)} points in the floor hull.")
    print(f"Detected {len(floor_corners)} corners in the floor hull.")

    # floor_lineset = create_lineset_from_contour(floor_lines)

    # # Show how many points are in the lineset
    # print(f"Lineset contains {len(floor_lineset.points)} points and {len(floor_lineset.lines)} lines.")

    # opce([floor_lineset])  # Display the lineset and hull

    # floor_hull_pcd = create_point_cloud(floor_hull)  # Green color for hull
    wall_hull_pcd = create_point_cloud(create_correct_height_wall_slice(floor_corners), color=[0, 1, 0])  # Green color for wall slice  # noqa: E501
    floor_corners_pcd = create_point_cloud(floor_corners, color=[1, 0, 0])  # Red color for corners
    wall_floor_merge = merge_pcds([floor_corners_pcd, wall_hull_pcd])
    # opce(wall_floor_merge)  # Display the merged point cloud with wall slice and floor corners

    new_ceiling_pcd = keep_ceiling_points_from_x_height(
        new_pcd_tuple[1],
        floor_corners_pcd,
        height=1.5
    )

    new_small_ceiling_pcd = filter_ceiling_points(new_ceiling_pcd, percentage_to_keep=0.02)

    all_merge = merge_pcds([wall_floor_merge, new_small_ceiling_pcd])
    opce(all_merge)

    # get_extent(floor_hull)

    # floor_corners = floor_corners[:(len(floor_corners) // 2)]

    # floor_corners_pcd = create_point_cloud(floor_corners)
    # opce(floor_corners_pcd)

    # wall_pcd = new_pcd_tuple[1]
    # ceiling_pcd = new_pcd_tuple[2]


if __name__ == "__main__":
    main()
