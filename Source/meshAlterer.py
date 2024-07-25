import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
import open3d as o3d
from tqdm import tqdm



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
        distance_threshold (float, optional): The maximum distance between a vertex in the mesh and the nearest point in the point cloud. Defaults to 0.05.
        visualize_mesh (bool, optional): Boolean to visualize the simplified mesh. Defaults to False.

    Raises:
        ValueError: If the mesh or the point cloud is empty.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The simplified mesh.
    """
    # Check if inputs are not empty
    if not mesh:
        raise ValueError("The mesh is empty.")
    
    if not original_point_cloud:
        raise ValueError("The point cloud is empty.")
    
    # Print that the simplification process has started
    print("Starting mesh simplification...")
    
    # Step 1: Get the number of triangles in the mesh
    triangle_amount = len(mesh.triangles)
    
    # Step 2: Compute the distance from each vertex in the mesh to the nearest point in the point cloud
    o_pcd_tree = o3d.geometry.KDTreeFlann(original_point_cloud)
    distances = []

    for vertex in tqdm(np.asarray(mesh.vertices), desc="Computing distances"):
        [_, idx, _] = o_pcd_tree.search_knn_vector_3d(vertex, 1)
        nearest_point = np.asarray(original_point_cloud.points)[idx[0]]
        distances.append(np.linalg.norm(vertex - nearest_point))

    distances = np.array(distances)

    # Step 3: Filter out the vertices and corresponding faces that are beyond the specified distance
    distance_threshold = 0.06  # Set your threshold here
    vertices_to_keep = distances <= distance_threshold

    # Create a new mesh with filtered vertices and faces
    new_vertices = np.asarray(mesh.vertices)[vertices_to_keep]
    new_colors = np.asarray(mesh.vertex_colors)[vertices_to_keep]
    vertex_map = {i: idx for idx, i in enumerate(np.where(vertices_to_keep)[0])}
    new_faces = []

    for face in tqdm(mesh.triangles, desc="Filtering faces"):
        if all(v in vertex_map for v in face):
            new_faces.append([vertex_map[v] for v in face])

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
        # Visualize the simplified mesh
        o3d.visualization.draw_geometries([smooth_mesh], mesh_show_back_face=True)
    
    print(f"Removed {triangle_amount - len(smooth_mesh.triangles)} triangles")
    
    return smooth_mesh


def transform_mesh_to_height_map(mesh: o3d.cpu.pybind.geometry.TriangleMesh, grid_size: int = 200, visualize_map: bool = False, debugging_logs: bool = False):
    """Transforms a mesh into a height map by projecting it onto the x-y plane.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be transformed into a height map.
        grid_size (int, optional): Number of grid points. Defaults to 200.
        visualize_map (bool, optional): Boolean to visualize the height map. Defaults to False.
        debugging_logs (bool, optional): Boolean to print debugging logs. Defaults to False.

    Returns:
        RegularGridInterpolator: An interpolator for the height map.
        cKDTree: A KDTree for the height map.
        np.ndarray: An array of valid points in the height map.
        np.ndarray: The height map grid.
        np.ndarray: X coordinates of the grid.
        np.ndarray: Y coordinates of the grid.
        float: The minimum height value.
        
    Raises:
        ValueError: If the mesh is empty.
    """    
    # Check if mesh is empty
    if not mesh:
        raise ValueError("The mesh is empty.")
    
    # Print that the transformation process has started
    print("Starting calculation of height map...")
        
    # Project mesh onto the x-y plane (use Z as height)
    vertices = np.asarray(mesh.vertices)
    x = vertices[:, 0]  # X-coordinates for positioning
    y = vertices[:, 1]  # Y-coordinates for positioning
    z = vertices[:, 2]  # Z-coordinates as height
    
    if debugging_logs:
        # Debugging: Print min and max values
        print("x range:", x.min(), x.max())
        print("y range:", y.min(), y.max())
        print("z range:", z.min(), z.max())

    # Create grid for contour map
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    z_min, z_max = z.min(), z.max()

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    # Show the amount of points in the grid
    if debugging_logs:
        print("Amount of points in the grid:", len(X.flatten()))
    
    
    # Keep per x-y coordinate the highest z value
    height_map = np.full((grid_size, grid_size), -np.inf)
    for i in tqdm(range(len(x)), desc="Creating height map"):
        x_idx = np.argmin(np.abs(x_grid - x[i]))
        y_idx = np.argmin(np.abs(y_grid - y[i]))
        height_map[x_idx, y_idx] = max(height_map[x_idx, y_idx], z[i])
        
    # So currently there can still be some holes in the height map, can be looked at later.
        
    # Show the amount of points in the height map
    if debugging_logs:
        print("Amount of points in the height map:", len(height_map.flatten()))
    
    # Visualize the height map
    if visualize_map:
        plt.imshow(height_map, cmap="viridis", norm=Normalize(vmin=z_min, vmax=z_max))
        plt.colorbar()
        plt.show()
        
    if debugging_logs:
        # Show statistics of the height map
        print("Height map statistics:")
        # # Print the first 10 rows of the height map
        # print(height_map[:10])
        print("Amount of points in the height map:", len(height_map.flatten()))
        print("Minimum height:", z_min)
        print("Maximum height:", z_max)
        print("Amount of valid points:", np.sum(height_map != -np.inf))
        print("Amount of invalid points:", np.sum(height_map == -np.inf))
        print("Amount of total points:", grid_size * grid_size)
        print("Percentage of valid points:", np.sum(height_map != -np.inf) / (grid_size * grid_size) * 100, "%")


def create_mesh_from_height_map(height_map: np.ndarray, X: np.ndarray, Y: np.ndarray) -> o3d.geometry.TriangleMesh:
    """Creates a new mesh from the height map.

    Args:
        height_map (np.ndarray): The height map grid.
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.

    Returns:
        o3d.geometry.TriangleMesh: The new mesh created from the height map.
    """
    # Create vertices from the height map
    vertices = []
    for i in tqdm(range(height_map.shape[0]), desc="Creating vertices"):
        for j in range(height_map.shape[1]):
            vertices.append([X[i, j], Y[i, j], height_map[i, j]])
    vertices = np.array(vertices)
    
    # Create triangles from the grid
    triangles = []
    for i in tqdm(range(height_map.shape[0] - 1), desc="Creating triangles"):
        for j in range(height_map.shape[1] - 1):
            idx = i * height_map.shape[1] + j
            triangles.append([idx, idx + 1, idx + height_map.shape[1]])
            triangles.append([idx + 1, idx + height_map.shape[1] + 1, idx + height_map.shape[1]])
    triangles = np.array(triangles)
    
    # Create the mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    return mesh

# Idea for next function:
# Cut out any point in the height map that at the bottom height
# After the mesh is cutout make a bottom layer under the height map so that the mesh is a closed volume
# Then repair any holes in the mesh so that it surelly is a closed volume
# Test out with higher resolution height map and mesh to see if the function works as intended

def drape_mesh_downward(
    mesh: o3d.geometry.TriangleMesh,
    interpolator: RegularGridInterpolator,
    tree: cKDTree,
    valid_points: np.ndarray,
    height_map: np.ndarray,
    X: np.ndarray,
    Y: np.ndarray,
    z_min: float,
    contour_distance: float = 0.1  # Distance threshold to determine contour relevance
    ) -> o3d.geometry.TriangleMesh:
    """Drapes a mesh downward to a height map, focusing on contour points.

    Args:
        mesh (o3d.geometry.TriangleMesh): The mesh to be draped.
        interpolator (RegularGridInterpolator): An interpolator for the height map.
        tree (cKDTree): A KDTree for the height map.
        valid_points (np.ndarray): An array of valid points in the height map.
        height_map (np.ndarray): The height map grid.
        X (np.ndarray): X coordinates of the grid.
        Y (np.ndarray): Y coordinates of the grid.
        z_min (float): The minimum height value.
        contour_distance (float): Distance from the contour within which to apply draping.

    Returns:
        o3d.geometry.TriangleMesh: The draped mesh.
    """
    # Print that the draping process has started
    print("Starting draping of mesh...")

    vertices = np.asarray(mesh.vertices)
    new_vertices = vertices.copy()

    # Create a set of contour points
    contour_points = set()
    
    for i in tqdm(range(len(X)), desc="Identifying contour points"):
        for j in range(len(Y)):
            if height_map[i, j] != z_min:
                contour_points.add((Y[i, j], X[i, j]))

    # Draping the mesh downward only within contour distance
    for i, vertex in enumerate(tqdm(vertices, desc="Draping mesh")):
        x, y = vertex[0], vertex[1]
        height_at_point = interpolator((y, x))

        if height_at_point == z_min:
            # Find the nearest valid height if no direct height is available
            distance, index = tree.query([y, x])
            if distance < np.inf:
                nearest_valid_point = valid_points[index]
                height_at_point = height_map[np.where(Y == nearest_valid_point[0])[0][0], np.where(X[0] == nearest_valid_point[1])[0][0]]

        # Check if the vertex is close to a contour point
        min_distance_to_contour = min(np.linalg.norm(np.array([x, y]) - np.array(contour_point)) for contour_point in contour_points)
        if min_distance_to_contour <= contour_distance:
            if height_at_point != -np.inf:
                new_vertices[i, 2] = min(vertex[2], height_at_point)

    # Create new mesh with draped vertices
    new_mesh = o3d.geometry.TriangleMesh()
    new_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    new_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.triangles))
    
    new_mesh.compute_vertex_normals()
    new_mesh.compute_triangle_normals()
    return new_mesh