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


def find_lines_in_pointcloud(pcd: o3d.geometry.PointCloud, alpha: float = 10) -> np.ndarray:
    """Find lines in a 3D point cloud by projecting to 2D and detecting boundaries.

    Args:
        pcd (o3d.geometry.PointCloud): The input point cloud.
        alpha (float, optional): The alpha parameter for the alpha shape. Defaults to 10.

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
        boundary = alpha_shape(points_2d, alpha)

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
    opce(new_pcd)

    floor_lines = find_lines_in_pointcloud(new_pcd_tuple[0])
    print(f"Detected {len(floor_lines)} lines in the floor point cloud.")  # Lies
    floor_pcd = create_point_cloud(floor_lines)
    opce(floor_pcd)

    floor_corners = sort_points_in_hull(floor_lines, 0.05)
    print(f"Detected {len(floor_corners)} corners in the floor lines.")  # Lies

    floor_corners = floor_corners[:(len(floor_corners) // 2)]

    floor_corners_pcd = create_point_cloud(floor_corners)
    opce(floor_corners_pcd)

    # wall_pcd = new_pcd_tuple[1]
    # ceiling_pcd = new_pcd_tuple[2]


if __name__ == "__main__":
    main()
