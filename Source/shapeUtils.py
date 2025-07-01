import copy

import numpy as np
import open3d as o3d
from tqdm import tqdm


def merge_pcd(
        pcd1: o3d.cpu.pybind.geometry.PointCloud, pcd2: o3d.cpu.pybind.geometry.PointCloud
) -> o3d.cpu.pybind.geometry.PointCloud:
    """Function to merge two point clouds into one.
    Past me: Idea to use a list of point clouds and then merge them all at once using some sort of loop.

    Args:
        pcd1 (o3d.cpu.pybind.geometry.PointCloud): Point cloud 1.
        pcd2 (o3d.cpu.pybind.geometry.PointCloud): Point cloud 2.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud 1 and 2 merged into one.
    """
    # Check if the input point clouds are empty
    if len(pcd1.points) == 0:
        raise ValueError("The first input point cloud is empty.")
    if len(pcd2.points) == 0:
        raise ValueError("The second input point cloud is empty.")

    # Check if the input point clouds have colours
    if not pcd1.has_colors():
        raise ValueError("The first input point cloud does not have colours.")
    if not pcd2.has_colors():
        raise ValueError("The second input point cloud does not have colours.")

    # Merge the points of the two point clouds
    p1_load = np.asarray(pcd1.points)
    p2_load = np.asarray(pcd2.points)
    p3_load = np.concatenate((p1_load, p2_load), axis=0)

    # The colours of the points are also merged below
    p1_color = np.asarray(pcd1.colors)
    p2_color = np.asarray(pcd2.colors)
    p3_color = np.concatenate((p1_color, p2_color), axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p3_load)
    pcd.colors = o3d.utility.Vector3dVector(p3_color)

    return pcd


def merge_list_of_pointclouds(pcd_list: list) -> o3d.cpu.pybind.geometry.PointCloud:
    """Function to merge a list of point clouds into one.

    Args:
        pcd_list (list): List of point clouds to be merged.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point clouds merged into one.
    """
    # Check if the input list is empty
    if len(pcd_list) == 0:
        raise ValueError("The input list of point clouds is empty.")

    # Check if the input list contains point clouds
    if not all(isinstance(pcd, o3d.cpu.pybind.geometry.PointCloud) for pcd in pcd_list):
        raise ValueError("The input list contains elements that are not point clouds.")

    # Check if the input list contains point clouds with points
    if not all(len(pcd.points) > 0 for pcd in pcd_list):
        # remove the point clouds with no points
        pcd_list = [pcd for pcd in pcd_list if len(pcd.points) > 0]

        # Check again if there are still point clouds in the list that have no points
        if not all(len(pcd.points) > 0 for pcd in pcd_list):
            raise ValueError("The input list contains point clouds with no points.")

    # Add each point cloud from the list to a new point cloud
    pcd = o3d.geometry.PointCloud()
    print("Merging point clouds...")
    for i in tqdm(range(len(pcd_list))):
        if i == 0:
            # If it's the first point cloud, just copy it
            pcd = copy.deepcopy(pcd_list[i])
        pcd = merge_pcd(pcd, pcd_list[i])

    print("Merging point clouds done.")
    print(f"Merged {len(pcd_list)} point clouds into one point cloud with {len(pcd.points)} points.")

    return pcd


def segment_plane(
    point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    visualize_plane: bool = False,
    visualize_leftovers: bool = False,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 10000,
    print_bool: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to extract a plane from a point cloud.
    Eyo this actually works, I'm so happy with this. I'm gonna use this for the next step in the pipeline.
    Maybe not exactly as I expected, but I can work with this. 🤩

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be processed.

        visualize_plane (bool, optional): A boolean to determine whether or not to visualize the extracted plane.
            Defaults to False.

        visualize_leftovers (bool, optional): A boolean to determine whether or not to visualize the points that are not in the
            extracted plane.
            Defaults to False.

        distance_threshold (float, optional): The maximum distance a point can be from the plane to be considered an inlier.
            Defaults to 0.01.

        ransac_n (int, optional): The number of points to sample for each iteration of RANSAC.
            Defaults to 3.

        num_iterations (int, optional): The number of iterations to run RANSAC.
            Defaults to 10000.

        print_bool (bool, optional): A boolean to toggle print statement the number of points in the extracted plane and the
            leftover points.
            Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The extracted plane in point cloud format.
        o3d.cpu.pybind.geometry.PointCloud: The points that are not in the extracted plane.
    """
    # Check if the input point cloud is empty
    if len(point_cloud.points) == 0:
        raise ValueError("The input point cloud is empty.")

    # Use RANSAC to segment the point cloud into planes
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    extracted_points = point_cloud.select_by_index(inliers)
    not_extracted_points = point_cloud.select_by_index(inliers, invert=True)

    if print_bool:
        print(f"Extracted a plane with {len(extracted_points.points)} points from the point cloud")
        print(f"Kept {len(not_extracted_points.points)} points in the point cloud")

    if visualize_plane:
        # Visualize the extracted plane
        o3d.visualization.draw_geometries([extracted_points])

    if visualize_leftovers:
        # Visualize the leftover points
        o3d.visualization.draw_geometries([not_extracted_points])

    # Return the extracted plane and the leftover points
    return extracted_points, not_extracted_points


def find_plane_module_manual(
    point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 10000
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

    Returns:
        open3d.geometry.PointCloud: The extracted plane equation.
    """
    # Check if the input point cloud is empty
    if len(point_cloud.points) == 0:
        raise ValueError("The input point cloud is empty.")

    pcd = point_cloud

    # Initialize variables
    current_plane = None
    previous_plane = None

    while True:
        # Segment the point cloud into a plane and the leftover points
        plane_pcd, leftover_pcd = segment_plane(pcd, False, False, distance_threshold, ransac_n, num_iterations)

        # Save the current plane equation as the new previous plane
        previous_plane = current_plane
        # Save the new plane equation as the current plane
        current_plane = plane_pcd

        # Copy and color the new found plane red
        copy_plane_pcd = copy.deepcopy(plane_pcd)
        coloured_plane_pcd = copy_plane_pcd.paint_uniform_color([1.0, 0, 0])
        visualize_list = [coloured_plane_pcd, leftover_pcd]

        # If there is a previous plane, color it green and add it to the list of point clouds to visualize
        if previous_plane is not None:
            copy_previous_plane = copy.deepcopy(previous_plane)
            coloured_previous_plane = copy_previous_plane.paint_uniform_color([0, 1.0, 0])
            # Add the previous plane to the list of point clouds to visualize
            visualize_list.append(coloured_previous_plane)

        # Visualize the current plane with inlier points
        o3d.visualization.draw_geometries(visualize_list, left=0, top=45)

        # Ask the user for input on whether to expand the plane or stop and return the current plane
        user_input = input("Enter 'e' to expand the plane, 'u' to undo the last expansion, 'p' to export the previous plane, 'r' to skip the current plane and find a new one or any other key to accept the current plane: ")  # noqa: E501
        if user_input == "e":
            # Expand the plane by removing inliers from the point cloud
            pcd = leftover_pcd
            if previous_plane is not None:
                # If there is a previous plane, add it to the current plane
                current_plane = merge_pcd(current_plane, previous_plane)

        # Maybe this is redundant, but I'm gonna keep it in here for now
        elif user_input == "u":
            # Undo the last expansion by restoring the previous plane and point cloud
            current_plane = previous_plane

        elif user_input == "p":
            # Keep the previous plane and return it as the output
            return previous_plane

        elif user_input == "r":
            # Retry and find a new plane
            current_plane = previous_plane
            pcd = leftover_pcd

        else:
            # Accept the current plane and return it as the output
            if previous_plane is not None:
                return merge_pcd(current_plane, previous_plane)
            return current_plane


def repair_point_cloud_module(
    input_point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    kdtree_radius: float = 0.1,
    kdtree_max_nn: int = 30,
    depth: int = 8,
    scale: float = 2.2,
    linear_fit: bool = True,
    quantile_value: float = 0.1,
    visualize: bool = False
) -> o3d.cpu.pybind.geometry.TriangleMesh:
    """A function that will try to reconstruct a point cloud by making a triangle mesh using the Screened Poisson Reconstruction.

    Args:
        input_point_cloud (o3d.cpu.pybind.geometry.PointCloud): Point cloud to reconstruct.
        kdtree_radius (float, optional): The radius to use for the KDTree search.
            So far altering this value doesn't seem to have any effect on the output.
            Defaults to 0.1.
        kdtree_max_nn (int, optional): The maximum number of nearest neighbors to use for the KDTree search.
            The higher the value the more accurate the reconstruction will be, but more faulty triangles will be generated.
            Defaults to 30.
        depth (int, optional): Maximum depth of the tree that will be used for surface reconstruction.
            Higher values will increase the runtime, but will also increase the quality of the reconstruction,
            but will be a slower and will overfit.
            Defaults to 8.
        scale (float, optional): Specifies the ratio between the diameter of the reconstruction cube and the diameter of the
            samples' bounding cube.
            Defaults to 2.2.
        linear_fit (bool, optional): If true, the reconstructor will use linear interpolation to estimate the positions of iso-
            vertices.
            Defaults to True.
        quantile_value (float, optional): The quantile value to use for removing the giant plane that will be generated by the
            Poisson reconstruction.
            Value must be between 0 and 1. The higher the value the more plane will be removed.
            Defaults to 0.1.
        visualize (bool, optional): A boolean that determines whether to visualize the reconstructed mesh or not.
            Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The reconstructed mesh.
    """
    # Check inputs
    # Check if the input point cloud is empty
    if len(input_point_cloud.points) == 0:
        raise ValueError("The input point cloud is empty.")

    # Check if the value of the quantile is between 0 and 1
    if not 0 <= quantile_value <= 1:
        raise ValueError("The quantile value must be between 0 and 1.")

    # Compute normals
    input_point_cloud.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=kdtree_radius,
            max_nn=kdtree_max_nn
        )
    )
    # Normalize the point cloud using normalize_normals
    input_point_cloud.normalize_normals()

    # Hey this actually works really well and fast 👀
    copy_pcd = copy.deepcopy(input_point_cloud)

    # Check if the point cloud actually has normals
    if not copy_pcd.has_normals():
        raise ValueError("The point cloud does not have normals, please calculate normals before running this function.")

    print("There might be an error that will pop up below :)")
    # Set the verbosity level of Open3D to only print severe errors
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    # Higher scale, more fill up, but also more random guessing and more "detail"
    # Pretty sure this line causes the warning about finding bad sample nodes
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        copy_pcd,
        depth=depth,
        scale=scale,
        linear_fit=linear_fit
    )

    # The code below will remove the giant plane that will be generated by the Poisson reconstruction
    # Note to future me: Maybe fiddle around with the quantile value to get better results 😘
    vertices_to_remove = densities < np.quantile(densities, quantile_value)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if visualize:
        # Show only the reconstructed mesh
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True, left=0, top=45)

    return mesh


def transform_mesh_to_pcd(
    mesh: o3d.cpu.pybind.geometry.TriangleMesh,
    original_pcd: o3d.cpu.pybind.geometry.PointCloud
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function that will transform a mesh to a point cloud.

    Args:
        mesh (o3d.cpu.pybind.geometry.TriangleMesh): The mesh to transform.
        original_pcd (o3d.cpu.pybind.geometry.PointCloud): The original point cloud that the mesh was generated from.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The transformed point cloud.
    """
    # Check if the input mesh is empty
    if len(mesh.vertices) == 0:
        raise ValueError("The input mesh is empty.")

    # Check if the input point cloud is empty
    if len(original_pcd.points) == 0:
        raise ValueError("The input point cloud is empty.")

    print("Transforming mesh to point cloud...")

    # calculate overall density of the point cloud
    original_point_mahalanobis_distance = o3d.geometry.PointCloud.compute_mahalanobis_distance(original_pcd)
    density = len(original_pcd.points) / np.sum(original_point_mahalanobis_distance)

    # Calculate roughly how many points the mesh will add compared to the original point cloud
    mesh_to_pcd = mesh.sample_points_uniformly(int(len(original_pcd.points) * density))

    return mesh_to_pcd
