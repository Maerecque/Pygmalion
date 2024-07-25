import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import numpy as np
import open3d as o3d
from scipy.interpolate import RegularGridInterpolator
from scipy.spatial import cKDTree
from scipy.ndimage import binary_dilation, binary_erosion
import open3d as o3d
from tqdm import tqdm

import pyvista as pv



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


def transform_mesh_to_height_map(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    grid_size: int = 200,
    visualize_map: bool = False,
    debugging_logs: bool = False):
    """Transforms a mesh into a height map by projecting it onto the x-y plane.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be transformed into a height map.
        grid_size (int, optional): Number of grid points. Defaults to 200.
        visualize_map (bool, optional): Boolean to visualize the height map. Defaults to False.
        debugging_logs (bool, optional): Boolean to print debugging logs. Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The point cloud representing the height map.
        
    Raises:
        ValueError: If the mesh is empty.
    """    
    # Check if mesh is empty
    if not mesh:
        raise ValueError("The mesh is empty.")
    
    # Print that the transformation process has started
    print("Starting calculation of height map...")
    
    # Print statistics of the mesh
    if debugging_logs:
        print("Mesh statistics:")
        print("Amount of vertices:", len(mesh.vertices))
        print("Amount of triangles:", len(mesh.triangles))
        
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
    
    # Visualize the height map
    if visualize_map:
        plt.imshow(height_map, cmap="viridis", norm=Normalize(vmin=z_min, vmax=z_max))
        plt.colorbar()
        plt.show()
        
    if debugging_logs:
        # Show statistics of the height map
        print("Height map statistics:")
        print("Amount of points in the height map:", len(height_map.flatten()))
        print("Minimum height:", z_min)
        print("Maximum height:", z_max)
        print("Amount of valid points:", np.sum(height_map != -np.inf))
        print("Amount of invalid points:", np.sum(height_map == -np.inf))
        print("Amount of total points:", grid_size * grid_size)
        print("Percentage of valid points:", np.sum(height_map != -np.inf) / (grid_size * grid_size) * 100, "%")
        
    # Transform the height map into an open3d point cloud to get an idea of the floorplan
    floor_plan_points = np.argwhere(height_map != -np.inf)
    floor_plan_coords = np.array([x_grid[floor_plan_points[:, 1]], y_grid[floor_plan_points[:, 0]], np.full(floor_plan_points.shape[0], z_min)]).T
    floor_plan_point_cloud = o3d.geometry.PointCloud()
    floor_plan_point_cloud.points = o3d.utility.Vector3dVector(floor_plan_coords)
    
    # Create the same pointcloud but with correct z values
    ceiling_points = np.argwhere(height_map != -np.inf)
    z_values = height_map[ceiling_points[:, 0], ceiling_points[:, 1]]
    ceiling_coords = np.array([x_grid[ceiling_points[:, 1]], y_grid[ceiling_points[:, 0]], z_values]).T
    ceiling_point_cloud = o3d.geometry.PointCloud()
    ceiling_point_cloud.points = o3d.utility.Vector3dVector(ceiling_coords)
    
    # Calculate the point density of the floor plan
    point_density = len(floor_plan_coords) / (grid_size * grid_size)

    if debugging_logs:
        print("Point density of the floor plan:", point_density)
    
    # Find the edges of the floor plan and ceiling based their x and y coordinates
    floor_plan_edges = np.argwhere(binary_dilation(height_map != -np.inf) ^ binary_erosion(height_map != -np.inf))
    ceiling_edges = np.argwhere(binary_dilation(height_map != -np.inf) ^ binary_erosion(height_map != -np.inf))
    
    # Visualize the floor plan and ceiling edges in a 3d plot
    if visualize_map:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(floor_plan_edges[:, 1], floor_plan_edges[:, 0], np.full(floor_plan_edges.shape[0], z_min), c='r', marker='o')
        ax.scatter(ceiling_edges[:, 1], ceiling_edges[:, 0], height_map[ceiling_edges[:, 0], ceiling_edges[:, 1]], c='b', marker='o')
        plt.show()
    
    # calculate the distance between all the floor plan edge points and the ceiling edge points and fill in the gaps with the same density as the floor plan
    # This is done to make sure the point cloud is complete
    # These new points will be called wall points
    # Calculate the wall points
    # Create walls between floor edges and ceiling edges
    wall_points = []

    for floor_edge in floor_plan_edges:
        floor_x, floor_y = x_grid[floor_edge[1]], y_grid[floor_edge[0]]
        corresponding_ceiling = next((ceiling_edge for ceiling_edge in ceiling_edges if np.array_equal(ceiling_edge[:2], floor_edge[:2])), None)
        if corresponding_ceiling is not None:
            ceiling_z = height_map[corresponding_ceiling[0], corresponding_ceiling[1]]
            floor_z = z_min
            height_difference = ceiling_z - floor_z
            if np.isfinite(height_difference) and height_difference > 0:
                num_points = int(height_difference / point_density) + 1
                for i in range(num_points + 1):
                    wall_z = floor_z + i * (height_difference / num_points)
                    wall_points.append([floor_x, floor_y, wall_z])

    wall_points = np.array(wall_points)

    if wall_points.size > 0:
        wall_point_cloud = o3d.geometry.PointCloud()
        wall_point_cloud.points = o3d.utility.Vector3dVector(wall_points)
    else:
        wall_point_cloud = o3d.geometry.PointCloud()
    
    # Combine the floor plan and ceiling point cloud
    hull_points = np.concatenate([
        np.asarray(floor_plan_point_cloud.points),
        np.asarray(ceiling_point_cloud.points),
        np.asarray(wall_point_cloud.points)
    ])
    hull_point_cloud = o3d.geometry.PointCloud()
    hull_point_cloud.points = o3d.utility.Vector3dVector(hull_points)
    
    # Visualize the point cloud
    if visualize_map:
        o3d.visualization.draw_geometries([hull_point_cloud])

    return hull_point_cloud
