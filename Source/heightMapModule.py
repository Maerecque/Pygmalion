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


def create_grid(x_range: tuple, y_range: tuple, nx: int, ny: int) -> tuple:
    """Create a grid for contour mapping.

    Args:
        x_range (tuple): Minimum and maximum x values.
        y_range (tuple): Minimum and maximum y values.
        nx (int): Number of grid points in x-direction.
        ny (int): Number of grid points in y-direction.

    Returns:
        tuple: Meshgrid of x and y coordinates, and arrays of x_grid and y_grid.
    """
    x_grid = np.linspace(x_range[0], x_range[1], nx)
    y_grid = np.linspace(y_range[0], y_range[1], ny)
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

    # Precompute scaling factors for direct index mapping
    x_min, x_max = x_grid[0], x_grid[-1]
    y_min, y_max = y_grid[0], y_grid[-1]
    x_step = (x_max - x_min) / (len(x_grid) - 1)
    y_step = (y_max - y_min) / (len(y_grid) - 1)

    # Compute grid indices for all points in one go
    x_idx = np.clip(((x - x_min) / x_step).astype(int), 0, len(x_grid) - 1)
    y_idx = np.clip(((y - y_min) / y_step).astype(int), 0, len(y_grid) - 1)

    # Update height map (keeping max z per cell)
    for xi, yi, zi in tqdm(zip(x_idx, y_idx, z), total=len(z), desc="Creating height map"):
        if zi > height_map[xi, yi]:
            height_map[xi, yi] = zi

    return height_map


def create_point_cloud(coords: np.ndarray, color: tuple = None) -> o3d.cpu.pybind.geometry.PointCloud:
    """Create an Open3D point cloud from coordinates, with optional uniform color.

    Args:
        coords (numpy.ndarray): Array of coordinates of shape (N, 3).
        color (tuple, optional): RGB color as a tuple (R, G, B) in [0, 1].

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Open3D point cloud object.
    """
    try:
        if not isinstance(coords, np.ndarray):
            if isinstance(coords, list):
                coords = np.vstack(coords)
            else:
                raise TypeError("Input coordinates must be a numpy array or a list of arrays.")
    except Exception as e:
        print(f"Error creating point cloud: {e}")
        return None

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(coords)

    if color is not None:
        if len(color) != 3 or not all(0 <= c <= 1 for c in color):
            raise ValueError("Color must be a tuple of 3 floats in range [0, 1].")
        colors = np.tile(color, (coords.shape[0], 1))
        point_cloud.colors = o3d.utility.Vector3dVector(colors)

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
    grid_spacing_m: float
):
    """Generate points between floor and ceiling edges to create walls.

    Args:
        floor_edges (numpy.ndarray): Array of floor edge coordinates.
        ceiling_edges (numpy.ndarray): Array of ceiling edge coordinates.
        height_map (numpy.ndarray): 2D array representing the height map.
        x_grid (numpy.ndarray): Grid of x-coordinates.
        y_grid (numpy.ndarray): Grid of y-coordinates.
        z_min (float): Minimum z-value.
        grid_spacing_m (float): Desired spacing (in meters) between wall points.

    Returns:
        numpy.ndarray: Array of wall points.
    """
    # Dictionary lookup for ceiling edges keyed by (row, col)
    ceiling_lookup = {(int(edge[0]), int(edge[1])): edge for edge in ceiling_edges}

    wall_points = []

    for fr, fc in tqdm(floor_edges[:, :2], desc="Creating walls"):
        key = (int(fr), int(fc))
        if key in ceiling_lookup:
            cr, cc = ceiling_lookup[key][:2]  # matching ceiling coords
            ceiling_z = height_map[int(cr), int(cc)]
            height_diff = ceiling_z - z_min

            if np.isfinite(height_diff) and height_diff > 0:
                num_points = int(height_diff / grid_spacing_m) + 1
                z_values = np.linspace(z_min, ceiling_z, num_points)
                x_val = x_grid[int(fr)]
                y_val = y_grid[int(fc)]
                wall_points.extend(np.column_stack((
                    np.full(z_values.shape, x_val),
                    np.full(z_values.shape, y_val),
                    z_values
                )))

    return np.array(wall_points)


def transform_pointcloud_to_height_map(
    pcd: o3d.geometry.PointCloud,
    grid_spacing_cm: float = 200,
    visualize_map: bool = False,
    visualize_map_np: bool = False,
    debugging_logs: bool = False
) -> tuple:
    """
    Transforms a point cloud into a height map by projecting it onto the x-y plane.

    Args:
        pcd (o3d.geometry.PointCloud): The point cloud to be transformed into a height map.
        grid_spacing_cm (float, optional): Grid spacing in centimeters. Defaults to 200.
        visualize_map (bool, optional): Boolean to visualize the height map. Defaults to False.
        visualize_map_np (bool, optional): Boolean to visualize the height map using numpy. Defaults to False.
        debugging_logs (bool, optional): Boolean to print debugging logs. Defaults to False.

    Returns:
        tuple: Three Open3D PointCloud objects representing the floor plan, ceiling, and wall points.

    Raises:
        ValueError: If the point cloud is empty.
    """
    if not pcd or len(pcd.points) == 0:
        raise ValueError("The point cloud is empty.")

    # Project points onto the x-y plane
    points = np.asarray(pcd.points)
    x, y, z = project_vertices_to_plane(points)

    # Note to self: in EPSG28992 is 1 heel getal 1 meter

    if debugging_logs:
        print(f"x range: {x.min()} to {x.max()}, y range: {y.min()} to {y.max()}, z range: {z.min()} to {z.max()}")

    # Convert cm spacing to meters
    grid_spacing_m = grid_spacing_cm / 100.0

    # Determine number of grid points from spacing
    nx = int(np.ceil((x.max() - x.min()) / grid_spacing_m)) + 1
    ny = int(np.ceil((y.max() - y.min()) / grid_spacing_m)) + 1

    # Create a grid for contour mapping
    (X, Y), x_grid, y_grid = create_grid((x.min(), x.max()), (y.min(), y.max()), nx, ny)

    # Generate a height map
    height_map = generate_height_map(x, y, z, x_grid, y_grid)

    if visualize_map and visualize_map_np:
        # Visualize the height map
        plt.imshow(height_map, cmap="viridis", norm=Normalize(vmin=z.min(), vmax=z.max()))
        plt.colorbar()
        plt.show()

    # Generate floor plan coordinates
    floor_plan_coords = np.column_stack([
        x_grid[np.argwhere(height_map != -np.inf)[:, 0]],
        y_grid[np.argwhere(height_map != -np.inf)[:, 1]],
        np.full(np.argwhere(height_map != -np.inf).shape[0], z.min())
    ])

    # Generate ceiling coordinates
    ceiling_coords = np.column_stack([
        x_grid[np.argwhere(height_map != -np.inf)[:, 0]],
        y_grid[np.argwhere(height_map != -np.inf)[:, 1]],
        height_map[np.argwhere(height_map != -np.inf)[:, 0], np.argwhere(height_map != -np.inf)[:, 1]]
    ])

    # Create floor plan point cloud
    floor_plan_point_cloud = create_point_cloud(floor_plan_coords)

    # Create ceiling point cloud
    ceiling_point_cloud = create_point_cloud(ceiling_coords)

    # Find edges of the height map
    floor_edges = find_edges(height_map)
    ceiling_edges = find_edges(height_map)

    # Generate wall points
    wall_points = generate_wall_points(floor_edges, ceiling_edges, height_map, x_grid, y_grid, z.min(), grid_spacing_m)

    # Create wall point cloud
    wall_point_cloud = create_point_cloud(wall_points) if wall_points.size > 0 else o3d.geometry.PointCloud()

    if visualize_map:
        # Visualize the floor plan, ceiling, and wall point clouds
        o3d.visualization.draw_geometries([floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud],
                                          mesh_show_back_face=True)

    return floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud


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

    if debugging_logs:
        print(f"Point density: {point_density:.2f} points per grid cell")

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
