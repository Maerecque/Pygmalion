import numpy as np
import open3d as o3d
from tqdm import tqdm
from scipy.ndimage import binary_dilation, binary_erosion
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


def project_vertices_to_plane(vertices: np.ndarray) -> tuple:
    """Project vertices onto the x-y plane.

    Args:
        vertices (numpy.ndarray): Array of vertex coordinates.

    Returns:
        tuple: Three numpy arrays representing x, y, and z coordinates.
    """
    x, y, z = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return x, y, z


def create_grid(x_range: tuple, y_range: tuple, grid_size: int) -> tuple:
    """Create a grid for contour mapping.

    Args:
        x_range (tuple): Minimum and maximum x values.
        y_range (tuple): Minimum and maximum y values.
        grid_size (int): Number of grid points.

    Returns:
        tuple: Meshgrid of x and y coordinates, and arrays of x_grid and y_grid.
    """
    x_grid = np.linspace(x_range[0], x_range[1], grid_size)
    y_grid = np.linspace(y_range[0], y_range[1], grid_size)
    return np.meshgrid(x_grid, y_grid), x_grid, y_grid


def generate_height_map(x: np.ndarray, y: np.ndarray, z: np.ndarray, x_grid: np.ndarray, y_grid: np.ndarray) -> np.ndarray:
    """Generate a height map by keeping the highest z value for each x-y coordinate.

    Args:
        x (numpy.ndarray): X-coordinates.
        y (numpy.ndarray): Y-coordinates.
        z (numpy.ndarray): Z-coordinates.
        x_grid (numpy.ndarray): Grid of x-coordinates.
        y_grid (numpy.ndarray): Grid of y-coordinates.

    Returns:
        numpy.ndarray: 2D array representing the height map.
    """
    height_map = np.full((len(x_grid), len(y_grid)), -np.inf)
    for i in tqdm(range(len(x)), desc="Creating height map"):
        x_idx = np.argmin(np.abs(x_grid - x[i]))
        y_idx = np.argmin(np.abs(y_grid - y[i]))
        height_map[x_idx, y_idx] = max(height_map[x_idx, y_idx], z[i])
    return height_map


def create_point_cloud(coords: np.ndarray) -> o3d.cpu.pybind.geometry.PointCloud:
    """Create an Open3D point cloud from coordinates.

    Args:
        coords (numpy.ndarray): Array of coordinates.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Open3D point cloud object.
    """
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords)
    return point_cloud


def find_edges(height_map: np.ndarray) -> np.ndarray:
    """Find the edges of the height map based on x and y coordinates.

    Args:
        height_map (numpy.ndarray): 2D array representing the height map.

    Returns:
        numpy.ndarray: Array of edge coordinates.
    """
    return np.argwhere(binary_dilation(height_map != -np.inf) ^ binary_erosion(height_map != -np.inf))


def generate_wall_points(
    floor_edges: np.ndarray,
    ceiling_edges: np.ndarray,
    height_map: np.ndarray,
    x_grid: np.ndarray,
    y_grid: np.ndarray,
    z_min: float,
    point_density: float
):
    """Generate points between floor and ceiling edges to create walls.

    Args:
        floor_edges (numpy.ndarray): Array of floor edge coordinates.
        ceiling_edges (numpy.ndarray): Array of ceiling edge coordinates.
        height_map (numpy.ndarray): 2D array representing the height map.
        x_grid (numpy.ndarray): Grid of x-coordinates.
        y_grid (numpy.ndarray): Grid of y-coordinates.
        z_min (float): Minimum z-value.
        point_density (float): Density of points.

    Returns:
        numpy.ndarray: Array of wall points.
    """
    wall_points = []
    for floor_edge in tqdm(floor_edges, desc="Creating walls"):
        floor_x, floor_y = x_grid[floor_edge[0]], y_grid[floor_edge[1]]
        corresponding_ceiling = next(
            (ceiling_edge for ceiling_edge in ceiling_edges if np.array_equal(ceiling_edge[:2], floor_edge[:2])), None
        )
        if corresponding_ceiling is not None:
            ceiling_z = height_map[corresponding_ceiling[0], corresponding_ceiling[1]]
            height_difference = ceiling_z - z_min
            if np.isfinite(height_difference) and height_difference > 0:
                num_points = int(height_difference / point_density) + 1
                wall_points.extend(
                    [[floor_x, floor_y, z_min + i * (height_difference / num_points)] for i in range(num_points + 1)]
                )
    return np.array(wall_points)


def transform_mesh_to_height_map(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    grid_size: int = 200,
    visualize_map: bool = False,
    debugging_logs: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """Transforms a mesh into a height map by projecting it onto the x-y plane.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to be transformed into a height map.
        grid_size (int, optional): Number of grid points. Defaults to 200.
        visualize_map (bool, optional): Boolean to visualize the height map. Defaults to False.
        debugging_logs (bool, optional): Boolean to print debugging logs. Defaults to False.

    Returns:
        tuple: Three Open3D PointCloud objects representing the floor plan, ceiling, and wall points.

    Raises:
        ValueError: If the mesh is empty.
    """
    if not mesh:
        raise ValueError("The mesh is empty.")

    # Project vertices onto the x-y plane
    vertices = np.asarray(mesh.vertices)
    x, y, z = project_vertices_to_plane(vertices)

    if debugging_logs:
        print(f"x range: {x.min()} to {x.max()}, y range: {y.min()} to {y.max()}, z range: {z.min()} to {z.max()}")

    # Create a grid for contour mapping
    (X, Y), x_grid, y_grid = create_grid((x.min(), x.max()), (y.min(), y.max()), grid_size)

    # Generate a height map
    height_map = generate_height_map(x, y, z, x_grid, y_grid)

    if visualize_map:
        # Visualize the height map
        plt.imshow(height_map, cmap="viridis", norm=Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar()
        plt.show()

    # Generate floor plan coordinates
    floor_plan_coords = np.column_stack([x_grid[np.argwhere(height_map != -np.inf)[:, 0]],
                                         y_grid[np.argwhere(height_map != -np.inf)[:, 1]],
                                         np.full(np.argwhere(height_map != -np.inf).shape[0], z.min())])

    # Generate ceiling coordinates
    ceiling_coords = np.column_stack([x_grid[np.argwhere(height_map != -np.inf)[:, 0]],
                                      y_grid[np.argwhere(height_map != -np.inf)[:, 1]],
                                      height_map[np.argwhere(height_map != -np.inf)[:, 0],
                                                 np.argwhere(height_map != -np.inf)[:, 1]]])

    # Create floor plan point cloud
    floor_plan_point_cloud = create_point_cloud(floor_plan_coords)

    # Create ceiling point cloud
    ceiling_point_cloud = create_point_cloud(ceiling_coords)

    # Calculate point density
    point_density = len(floor_plan_coords) / (grid_size * grid_size)

    # Find edges of the height map
    floor_edges = find_edges(height_map)
    ceiling_edges = find_edges(height_map)

    # Generate wall points
    wall_points = generate_wall_points(floor_edges, ceiling_edges, height_map, x_grid, y_grid, z.min(), point_density)

    # Create wall point cloud
    wall_point_cloud = create_point_cloud(wall_points) if wall_points.size > 0 else o3d.geometry.PointCloud()

    if visualize_map:
        # Visualize the floor plan, ceiling, and wall point clouds
        o3d.visualization.draw_geometries([floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud],
                                          mesh_show_back_face=True)

    return floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud
