import open3d as o3d
import numpy as np


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


def find_alpha_shapes(point_cloud: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.TriangleMesh:
    """A function to find the alpha shapes of a point cloud.
    DON'T WORK :(
    If it don't work then why it here?!

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be processed.

    Returns:
        o3d.cpu.pybind.geometry.TriangleMesh: The alpha shape of the point cloud.
    """
    # Compute alpha shape
    alpha = 0.5  # This value determines the shape of the alpha shape. Higher values produce more convex shapes.
    alpha_shape = o3d.geometry.AlphaShape(point_cloud, alpha)
    mesh = alpha_shape.get_mesh()

    # Get exterior points
    exterior_points = []
    for point in point_cloud.points:
        if not mesh.is_inside(point):
            exterior_points.append(point)

    return mesh


def find_outside_pointcloud(point_cloud: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to find the points that are outside the convex hull of a point cloud.
    !!DOES NOT WORK AS INTENDED!!
    With does not work as intended I meant: It works, but the shape of the convex hull is too basic, so it doesn't work for complex shapes
    *(which are the scans we use for this)* 🤡
    So again if it don't work then why it here?!

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be processed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The points that are outside the convex hull of the point cloud.
    """
    # Compute the convex hull of the point cloud
    hull, _ = point_cloud.compute_convex_hull(joggle_inputs=True)  # Joggle_inputs doesn't seem to do anything...

    # Get the vertices of the convex hull
    hull_vertices = hull.vertices

    print(type(hull_vertices))

    # visualize the convex hull
    o3d.visualization.draw_geometries([hull])

    # Select only the points that are outside the convex hull
    exterior_points = []
    for point in point_cloud.points:
        is_inside = False
        for vertex in hull_vertices:
            if point.all() == vertex.all():
                is_inside = True
        if not is_inside:
            print("outer_point_found")
            exterior_points.append(point)

    print(exterior_points)

    return exterior_points


def keep_points_in_view(point_cloud: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.PointCloud:
    """_summary_
    I still don't know how to get this function to work.
    For the last time, if it don't work then why it here?! 🤡

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): _description_

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: _description_
    """
    diameter = np.linalg.norm(
        np.asarray(point_cloud.get_max_bound()) - np.asarray(point_cloud.get_min_bound())
    )

    print(point_cloud.get_max_bound())

    # Mostly it's just this camera feature that I just don't understand. I might look back at this another day
    camera = [0, 0, diameter]
    radius = diameter * 100

    # This gives a bit of information about the function, but not enough to fricking understand it or atleast get how to get Camera to do what I want.
    # Maybe I should just look at the source code of the function, because ChatGPT is not helping me at all.
    help(o3d.cpu.pybind.geometry.PointCloud.hidden_point_removal)

    print("Get all points that are visible from given view point")
    _, pt_map = point_cloud.hidden_point_removal(camera, radius)

    # print(_)
    # print(type(_))

    # o3d.visualization.draw_geometries([pt_map])

    pcd = point_cloud.select_by_index(pt_map)
    o3d.visualization.draw_geometries([pcd])


def detect_planar_patches(point_cloud: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.PointCloud:
    # I uhh don't know what this is or why it's here, but I'm not gonna delete it because I might need it later.
    # **** Man I'm so tired of this, my onsetting alzheimers is really messing this up****

    # Estimate normals for the point cloud
    point_cloud.estimate_normals()

    # Define the distance threshold for plane segmentation
    distance_threshold = 0.01

    # Define the minimum number of points required for a plane
    min_inliers = 1000

    # My brother in christ, I will just call on this to find out what I can do with this function
    help(o3d.cpu.pybind.geometry.PointCloud)

    # Use RANSAC to segment the point cloud into planes
    # Planar patches exists in documentation online, but not in the help function or in the source code.
    planes = o3d.geometry.PointCloud.detect_planar_patches(point_cloud, distance_threshold, min_inliers)

    # Print the number of planes found
    print(f"Found {len(planes)} planes")

    # Visualize the point cloud with the segmented planes
    o3d.visualization.draw_geometries([point_cloud, *planes])


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


def expand_plane(point_cloud: o3d.cpu.pybind.geometry.PointCloud):
    # A function that will take a point cloud and allow the user to extract a plane from it.
    # By user-input the user can decide to expand the plane with the segment_plane function or to stop and return the plane that was last accepted.
    # The user can also decide to undo the last expansion and return to the previous plane.
    """
    Extracts a plane from a point cloud based on user input.

    Args:
        pcd (open3d.geometry.PointCloud): The point cloud to extract the plane from.

    Returns:
        open3d.geometry.PlaneEquation: The extracted plane equation.
    """
    pcd = point_cloud

    # Initialize variables
    current_plane = None
    previous_plane = None

    while True:
        # Segment the point cloud into a plane and the leftover points
        plane_pcd, leftover_pcd = segment_plane(pcd)

        # Save the current plane equation as the new previous plane
        previous_plane = current_plane
        # Save the new plane equation as the current plane
        current_plane = plane_pcd

        # Color the new found plane red
        coloured_plane_pcd = plane_pcd.paint_uniform_color([1.0, 0, 0])
        visualize_list = [coloured_plane_pcd, leftover_pcd]

        # If there is a previous plane, color it green and add it to the list of point clouds to visualize
        if previous_plane is not None:
            coloured_previous_plane = previous_plane.paint_uniform_color([0, 1.0, 0])
            # Add the previous plane to the list of point clouds to visualize
            visualize_list.append(coloured_previous_plane)

        # Visualize the current plane with inlier points
        o3d.visualization.draw_geometries(visualize_list)

        # Ask the user for input on whether to expand the plane or stop and return the current plane
        user_input = input("Enter 'e' to expand the plane, 'u' to undo the last expansion, 'p' to export the previous plane or any other key to accept the current plane: ")  # noqa: E501
        if user_input == "e":
            # Expand the plane by removing inliers from the point cloud
            pcd = leftover_pcd
            if previous_plane is not None:
                # If there is a previous plane, add it to the current plane
                current_plane = merge_pcd(current_plane, previous_plane)
        elif user_input == "u":
            # Undo the last expansion by restoring the previous plane and point cloud
            current_plane = previous_plane
        elif user_input == "p":
            # Keep the previous plane and return it as the output
            return previous_plane
        else:
            # Accept the current plane and return it as the output
            return current_plane
