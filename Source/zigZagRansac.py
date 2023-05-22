# In this file I will attempt to implement the following:
# 1. Make a function that will devide a pointcloud after normalization up into a 3d grid. The grid cells will have overlapping points with each other.
# 2. This function will run RANSAC on each grid cell. RANSAC will only keep planes that have a certain number of points.
# 3. After RANSAC is done, the function will return a list of planes that are in the pointcloud.

# Importing libraries
import numpy as np
import open3d as o3d
import fileHandler as fh
import shape_utils as su
from tqdm import tqdm


def divide_pointcloud_into_grid(
    pointcloud: o3d.geometry.PointCloud,
    grid_size: float,
    overlap: int
) -> dict:
    """A function to divide a pointcloud into a 3d grid.

    Args:
        pointcloud (o3d.geometry.PointCloud): Pointcloud to be divided.
        grid_size (float): Size of the grid cells.
        overlap (int): Overlap between the grid cells.

    Returns:
        dict : Dictionary containing the indices of the points in each cell.
    """
    # Get the minimum and maximum coordinates of the pointcloud
    min_bound = pointcloud.get_min_bound()
    max_bound = pointcloud.get_max_bound()

    # Calculate the number of cells in each dimension
    num_cells = np.ceil((max_bound - min_bound) / grid_size).astype(int)

    # Create an empty grid dictionary to store the divided point indices
    grid = {}

    print("Dividing pointcloud into a grid...")

    # Iterate over each point in the pointcloud
    for i, point in enumerate(tqdm(pointcloud.points)):
        # Calculate the cell index for the current point
        cell_index = np.floor((point - min_bound) / grid_size).astype(int)

        # Iterate over the neighboring cells in each dimension
        for dx in range(-overlap, overlap + 1):
            for dy in range(-overlap, overlap + 1):
                for dz in range(-overlap, overlap + 1):
                    neighbor_index = cell_index + np.array([dx, dy, dz])

                    # Check if the neighbor cell is within the grid bounds
                    if np.all(neighbor_index >= 0) and np.all(neighbor_index < num_cells):
                        # Convert the cell index to a string key
                        key = str(neighbor_index)

                        # Add the current point index to the corresponding cell in the grid
                        if key not in grid:
                            grid[key] = []
                        grid[key].append(i)

    # Print the number of cells in the grid
    print(f"The pointcloud has been divided into a {num_cells[0]} x {num_cells[1]} x {num_cells[2]} grid.")
    print("Number of cells in the grid: ", len(grid))

    return grid


def get_points_from_grid(
    pointcloud: o3d.geometry.PointCloud,
    grid: dict,
    cell_index: np.ndarray
) -> o3d.geometry.PointCloud:
    """A function to get the points from a cell in the grid.

    Args:
        pointcloud (o3d.geometry.PointCloud): Original pointcloud.
        grid (dict): Grid containing the indices of the points in each cell.
        cell_index (np.ndarray): Index of the cell to extract points from.

    Returns:
        o3d.geometry.PointCloud: Pointcloud containing the points from the cell.
    """
    # Get the indices of the points in the cell
    key = str(cell_index)
    point_indices = grid[key]

    # Get the points from the pointcloud
    points = np.asarray(pointcloud.points)[point_indices]

    # Return the points as pointcloud
    extracted_points = o3d.geometry.PointCloud()
    extracted_points.points = o3d.utility.Vector3dVector(points)

    # Get the colors from the pointcloud
    colors = np.asarray(pointcloud.colors)[point_indices]
    extracted_points.colors = o3d.utility.Vector3dVector(colors)

    # Return the extracted points
    return extracted_points


def ransac_plane_finder(
    point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 100000,
    min_plane_points: int = 500,
    print_found_planes: bool = False,
) -> o3d.cpu.pybind.geometry.PointCloud:
    """
    Extracts a plane from a point cloud based on user input.
    Note: This function is kind of slow with large point clouds, but it works.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to extract the plane from.

        distance_threshold (float, optional): The maximum distance a point can be from the plane to be considered an inlier.
            Defaults to 0.01.

        ransac_n (int, optional): The number of points to sample for each iteration of RANSAC.
            Defaults to 3.

        num_iterations (int, optional): The number of iterations to run RANSAC.
            Defaults to 10000.

        min_plane_points (int, optional): The minimum number of points a plane must have to be considered a plane.
            Defaults to 500.

        print_found_planes (bool, optional): A boolean to toggle the printing of found planes.
            Defaults to False.

    Returns:
        open3d.geometry.PointCloud: The extracted plane equation.
    """
    pcd = point_cloud

    # Initialize variables
    current_plane = None
    previous_plane = None

    amount_planes = 0

    for i in range(4):
        # Check if there are enough points left to do RANSAC.
        if len(pcd.points) < 3:
            # print("Not enough points left to do RANSAC.")
            continue

        # Segment the point cloud into a plane and the leftover points
        plane_pcd, leftover_pcd = su.segment_plane(pcd, False, False, distance_threshold, ransac_n, num_iterations)

        if len(plane_pcd.points) >= min_plane_points:
            if print_found_planes:
                print(f"- Found a plane with {len(plane_pcd.points)} points.")
            amount_planes += 1

            # Save the current plane equation as the new previous plane
            previous_plane = current_plane
            # Save the new plane equation as the current plane
            current_plane = plane_pcd

            pcd = leftover_pcd
            if previous_plane is not None:
                current_plane = su.merge_pcd(current_plane, previous_plane)

        # If the plane that was found is too small, skip it and continue with the next plane.
        else:
            if print_found_planes:
                print(f"- Found a plane with {len(plane_pcd.points)} points, which is too small.")
            current_plane = previous_plane
            pcd = plane_pcd
            # break

    if amount_planes > 0 and print_found_planes:
        print(f"Found {amount_planes} planes in the current cell.")

    else:
        if print_found_planes:
            print("- No planes found in the current cell.")

    if previous_plane is not None:
        return su.merge_pcd(current_plane, previous_plane)

    if current_plane is not None:
        return current_plane

    return None


def walk_through_grid(
    pointcloud: o3d.geometry.PointCloud,
    grid: dict,
    min_cell_size: int = 500,
    min_plane_points: int = 500
):
    """A function to walk through the grid and extract planes from each cell.

    Args:
        pointcloud (o3d.geometry.PointCloud): Original pointcloud.
        grid (dict): Grid containing the indices of the points in each cell.
        min_cell_size (int, optional): Minimum number of points in a cell to be considered for plane extraction.
            Defaults to 500.
        min_plane_points (int, optional): Minimum number of points in a plane to be considered a plane.
            Defaults to 500.
    """
    # Make an empty list to store the planes in
    planes = []

    print("Applying RANSAC over the grid...")

    # Iterate over each cell in the grid
    for key in tqdm(grid):
        if len(grid[key]) < min_cell_size:
            continue

        else:
            # print(key, len(grid[key]))

            # Get the points from the current cell
            grid_cell = get_points_from_grid(pointcloud, grid, np.array(key))

            # Extract a plane from the current cell
            # This will try to find a maximum of 4 planes in each cell with a minimum of 500 points in each plane
            plane_from_cell = ransac_plane_finder(grid_cell, 0.01, 3, 10000, min_plane_points)

            # Add the plane to the list of planes if it is not empty
            if plane_from_cell is not None:
                planes.append(plane_from_cell)

    # Merge all the planes into one pointcloud
    merged_planes = su.merge_list_of_pointclouds(planes)

    return merged_planes


if __name__ == "__main__":
    # Get the path to the LAS/LAZ file
    filename = fh.get_file_path("Select a LAS or LAZ file", ["*.las", "*.laz"])

    # Read the LAS/LAZ file
    pointcloud = fh.readout_LAS_file(filename)

    # Divide the pointcloud into a 3d grid
    grid = divide_pointcloud_into_grid(pointcloud, 1, 0)

    plane_pointcloud = walk_through_grid(pointcloud, grid)
