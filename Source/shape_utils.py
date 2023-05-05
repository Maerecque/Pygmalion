import open3d as o3d
import numpy as np
import copy

# Set the verbosity level of Open3D to only print severe errors
o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)


def merge_pcd(pcd1: o3d.cpu.pybind.geometry.PointCloud, pcd2: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.PointCloud:
    """Function to merge two point clouds into one.
    Past me: Idea to use a list of point clouds and then merge them all at once using some sort of loop.

    Args:
        pcd1 (o3d.cpu.pybind.geometry.PointCloud): Point cloud 1.
        pcd2 (o3d.cpu.pybind.geometry.PointCloud): Point cloud 2.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud 1 and 2 merged into one.
    """
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


def segment_plane(
    point_cloud: o3d.cpu.pybind.geometry.PointCloud,
    visualize_plane: bool = False,
    visualize_leftovers: bool = False,
    distance_threshold: float = 0.01,
    ransac_n: int = 3,
    num_iterations: int = 10000,
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to extract a plane from a point cloud.
    Eyo this actually works, I'm so happy with this. I'm gonna use this for the next step in the pipeline.
    Maybe not exactly as I expected, but I can work with this. 🤩

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be processed.

        visualize_plane (bool, optional): A boolean to determine whether or not to visualize the extracted plane.
            Defaults to False.

        visualize_leftovers (bool, optional): A boolean to determine whether or not to visualize the points that are not in the extracted plane.
            Defaults to False.

        distance_threshold (float, optional): The maximum distance a point can be from the plane to be considered an inlier.
            Defaults to 0.01.

        ransac_n (int, optional): The number of points to sample for each iteration of RANSAC.
            Defaults to 3.

        num_iterations (int, optional): The number of iterations to run RANSAC.
            Defaults to 10000.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The extracted plane in point cloud format.
        o3d.cpu.pybind.geometry.PointCloud: The points that are not in the extracted plane.
    """
    # Use RANSAC to segment the point cloud into planes
    plane_model, inliers = point_cloud.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=ransac_n,
        num_iterations=num_iterations
    )

    extracted_points = point_cloud.select_by_index(inliers)
    not_extracted_points = point_cloud.select_by_index(inliers, invert=True)

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


def expand_plane(
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
        user_input = input("Enter 'e' to expand the plane, 'u' to undo the last expansion, 'p' to export the previous plane or any other key to accept the current plane: ")  # noqa: E501
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


def create_alpha_shape(
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
            Higher values will increase the runtime, but will also increase the quality of the reconstruction, but will be a slower and will overfit.
            Defaults to 8.
        scale (float, optional): Specifies the ratio between the diameter of the reconstruction cube and the diameter of the samples' bounding cube.
            Defaults to 2.2.
        linear_fit (bool, optional): If true, the reconstructor will use linear interpolation to estimate the positions of iso-vertices.
            Defaults to True.
        quantile_value (float, optional): The quantile value to use for removing the giant plane that will be generated by the Poisson reconstruction.
            Value must be between 0 and 1. The higher the value the more plane will be removed.
            Defaults to 0.1.
        visualize (bool, optional): A boolean that determines whether to visualize the reconstructed mesh or not.
            Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The reconstructed mesh.
    """
    # Hey this actually works really well and fast 👀
    copy_cpd = copy.deepcopy(input_point_cloud)

    # Normalize the point cloud using normalize_normals
    o3d.geometry.PointCloud.estimate_normals(copy_cpd, search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=kdtree_radius, max_nn=kdtree_max_nn))
    copy_cpd.normalize_normals()

    # Higher scale, more fill up, but also more random guessing and more "detail"
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(copy_cpd, depth=depth, scale=scale, linear_fit=linear_fit)

    # The code below will remove the giant plane that will be generated by the Poisson reconstruction
    # Note to future me: Maybe fiddle around with the quantile value to get better results 😘
    vertices_to_remove = densities < np.quantile(densities, quantile_value)
    mesh.remove_vertices_by_mask(vertices_to_remove)

    if visualize:
        # Show original point cloud and the reconstructed mesh
        o3d.visualization.draw_geometries([mesh, input_point_cloud], mesh_show_back_face=True, left=0, top=45)

    return mesh
