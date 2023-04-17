import open3d as o3d
import numpy as np


def find_alpha_shapes(point_cloud: o3d.cpu.pybind.geometry.PointCloud) -> o3d.cpu.pybind.geometry.TriangleMesh:
    """A function to find the alpha shapes of a point cloud.
    DON'T WORK :(

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

    Args:
        point_cloud (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be processed.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: The points that are outside the convex hull of the point cloud.
    """
    # Compute the convex hull of the point cloud
    hull, _ = point_cloud.compute_convex_hull()

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
