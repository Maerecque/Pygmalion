import numpy as np
import open3d as o3d
from tqdm import tqdm
import pyvista as pv


def compute_distances_to_point_cloud(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    point_cloud: o3d.cpu.pybind.geometry.PointCloud
) -> np.ndarray:
    """Compute the distance from each vertex in the mesh to the nearest point in the point cloud.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh whose vertices' distances are computed.
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The original point cloud.

    Returns:
        numpy.ndarray: Array of distances for each vertex in the mesh.
    """
    kd_tree = o3d.geometry.KDTreeFlann(point_cloud)
    distances = []

    for vertex in tqdm(np.asarray(mesh.vertices), desc="Computing distances for down-sampling"):
        _, idx, _ = kd_tree.search_knn_vector_3d(vertex, 1)
        nearest_point = np.asarray(point_cloud.points)[idx[0]]
        distances.append(np.linalg.norm(vertex - nearest_point))

    return np.array(distances)


def filter_vertices_and_faces(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    distances: np.ndarray,
    distance_threshold: float
) -> tuple:
    """Filter out vertices and corresponding faces that are beyond the specified distance.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be filtered.
        distances (numpy.ndarray): Array of distances for each vertex in the mesh.
        distance_threshold (float): Maximum allowed distance for vertices to be kept.

    Returns:
        tuple: Arrays of new vertices, colors, and faces after filtering.
    """
    vertices_to_keep = distances <= distance_threshold
    new_vertices = np.asarray(mesh.vertices)[vertices_to_keep]
    new_colors = np.asarray(mesh.vertex_colors)[vertices_to_keep]
    vertex_map = {i: idx for idx, i in enumerate(np.where(vertices_to_keep)[0])}
    new_faces = [
        [vertex_map[v] for v in face]
        for face in tqdm(mesh.triangles, desc="Filtering faces for down-sampling")
        if all(v in vertex_map for v in face)
    ]

    return new_vertices, new_colors, new_faces


def mesh_simple_downsample(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    original_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    distance_threshold: float = 0.05,
    visualize_mesh: bool = False,
) -> o3d.cpu.pybind.geometry.TriangleMesh:
    """Simplifies a mesh by removing vertices that are far from the original point cloud.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be simplified.
        original_point_cloud (o3d.cpu.pybind.geometry.PointCloud): The original point cloud.
        distance_threshold (float, optional): Maximum distance between a vertex in the mesh and
            the nearest point in the point cloud. Defaults to 0.05.
        visualize_mesh (bool, optional): Whether to visualize the simplified mesh. Defaults to False.

    Raises:
        ValueError: If the mesh or the point cloud is empty.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The simplified mesh.
    """
    if not mesh:
        raise ValueError("The mesh is empty.")

    if not original_point_cloud:
        raise ValueError("The point cloud is empty.")

    print("Starting mesh simplification...")

    # Step 1: Compute distances from mesh vertices to the point cloud
    distances = compute_distances_to_point_cloud(mesh, original_point_cloud)

    # Step 2: Filter vertices and faces based on distance threshold
    new_vertices, new_colors, new_faces = filter_vertices_and_faces(mesh, distances, distance_threshold)

    # Step 3: Create a new mesh with filtered vertices and faces
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.vertex_colors = o3d.utility.Vector3dVector(new_colors)
    new_mesh.triangles = o3d.utility.Vector3iVector(new_faces)

    # Optional: Recompute normals for the new mesh
    new_mesh.compute_vertex_normals()
    new_mesh.compute_triangle_normals()

    # Smoothing the mesh
    smooth_mesh = new_mesh.filter_smooth_simple(number_of_iterations=15)

    if visualize_mesh:
        o3d.visualization.draw_geometries([smooth_mesh], mesh_show_back_face=True)

    print(f"Removed {len(mesh.triangles) - len(smooth_mesh.triangles)} triangles")

    return smooth_mesh


def transform_pcd_to_mesh(
    hull_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    alpha: float = 0.1,
    tolerance: float = 0.05,
    offset: float = 0.0,
    visualize_bool: bool = False,
    bool_3d_mesh: bool = True
) -> pv.UnstructuredGrid:
    """Transforms a point cloud into a mesh using the Delaunay algorithm.

    Args:
        hull_point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be transformed into a mesh.
        alpha (float, optional): Alpha value for the Delaunay algorithm. Defaults to 0.1.
        tolerance (float, optional): Tolerance value for the Delaunay algorithm. Defaults to 0.05.
        offset (float, optional): Offset value for the Delaunay algorithm. Defaults to 0.0.
        visualize_bool (bool, optional): Whether to visualize the mesh. Defaults to False.
        bool_3d_mesh (bool, optional): Whether to create a 3D mesh (True) or a 2D mesh (False). Defaults to True.

    Returns:
        pv.UnstructuredGrid: The mesh created from the point cloud.
    """
    # Convert the point cloud to a numpy array
    whole_cloud_points = np.asarray(hull_point_cloud.points)

    # Create a PyVista PolyData object from the point cloud
    cloud = pv.PolyData(whole_cloud_points)

    # Apply the Delaunay algorithm to create a mesh
    if bool_3d_mesh:
        volume = cloud.delaunay_3d(alpha=alpha, progress_bar=True, tol=tolerance, offset=offset)
    else:
        volume = cloud.delaunay_2d(alpha=alpha, progress_bar=True, tol=tolerance, offset=offset)

    # Visualize the resulting mesh if requested
    if visualize_bool:
        shell = volume.extract_geometry(progress_bar=True)
        shell.plot()

    return volume
