import open3d as o3d
import numpy as np
import sys
import os
from shapely.geometry import Polygon
from shapely import BufferJoinStyle

sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.linesetTools import contour_to_lineset, lineset_to_trianglemesh
from Source.pointCloudEditor import (  # noqa: F401
    open_point_cloud_editor as opce,
    open_mesh_and_lineset_viewer as omalv
)
from Source.pointCloudAltering import merge_point_clouds as merge_pcds


def expand_boundary(
    mesh: o3d.geometry.TriangleMesh,
    expand_distance: float = 0.0,
    point_visualization: bool = False
):
    """A function to expand the boundary of a mesh by a specified distance.
    This is done by converting the mesh vertices into a 2D polygon, expanding the polygon using Shapely's buffer function,
    and then converting the expanded polygon back into a mesh.

    NOTE: This function is currently only suitable for a flat 3D mesh.

    Args:
        mesh (o3d.geometry.TriangleMesh): The input Open3D mesh.
        expand_distance (float): The distance by which to expand the boundary. This has to be a positive value.
            This value is expressed in the same units as the mesh's vertices.
        point_visualization (bool): Whether to visualize the original and expanded points together. Defaults to False.

    Returns:
        o3d.geometry.TriangleMesh: A new Open3D mesh with the expanded boundary.
    """
    # Show amount of points in the original mesh (for debugging purposes)
    original_vertices = np.asarray(mesh.vertices)
    print("Number of original vertices:", len(original_vertices))
    # Get the z-value of the original vertices (assuming all vertices have the same z-value)
    z_value = original_vertices[0][2]  # Assuming all vertices have the same z-value, we can take it from the first vertex
    # Extract x and y coordinates as a list of sets, [(x, y), ...   ]
    twodee_coords = [(vertex[0], vertex[1]) for vertex in original_vertices]

    polygon = Polygon(twodee_coords)
    expanded_polygon = polygon.buffer(
        expand_distance,
        join_style=BufferJoinStyle.mitre,
    )
    expanded_2d_coords = list(expanded_polygon.exterior.coords)
    expanded_3d_coords = np.array([(x, y, z_value) for x, y in expanded_2d_coords])

    # Transform the expanded 3D coordinates back into an Open3D point cloud
    expanded_pcd = o3d.geometry.PointCloud()
    expanded_pcd.points = o3d.utility.Vector3dVector(expanded_3d_coords)

    if point_visualization:
        # Make a temporary point cloud of the original vertices for visualization purposes
        original_pcd = o3d.geometry.PointCloud()
        original_pcd.points = o3d.utility.Vector3dVector(original_vertices)

        cm_value = expand_distance * 100  # Convert to centimeters

        if np.isclose(cm_value, round(cm_value)):
            print(f"Uitbreidingsafstand: {int(round(cm_value))} cm")
        else:
            print(f"Uitbreidingsafstand: {cm_value:.2f} cm")

        # Visualize the original and expanded points together
        opce(merge_pcds([original_pcd, expanded_pcd]), show_help=False)
        # empty the original point cloud to free up memory
        original_pcd.clear()
        cm_value = None  # Clear the cm_value variable to free up memory

    expanded_lineset = contour_to_lineset(expanded_3d_coords)
    expanded_mesh = lineset_to_trianglemesh(expanded_lineset, expanded_3d_coords)
    expanded_pcd.clear()  # Clear the point cloud to free up memory
    expanded_lineset.clear()  # Clear the lineset to free up memory

    return expanded_mesh


def temp_READOUT_FUNCTION(file_path):
    """A temporary function to read out the point cloud from a ply file. This is just for testing purposes and will be replaced by the actual file loading function later on."""  # noqa: E501
    pcd = o3d.io.read_point_cloud(file_path)
    return pcd


if __name__ == "__main__":
    pcd = temp_READOUT_FUNCTION("C:/Users/marcz/3D Objects/Werfkelders/vloergrens.ply")
    pcd_points = np.asarray(pcd.points)
    pcd_lineset = contour_to_lineset(pcd_points)
    mesh = lineset_to_trianglemesh(pcd_lineset, pcd_points)

    # Expand the boundary of the mesh
    expanded_mesh = expand_boundary(mesh, expand_distance=0.3, point_visualization=True)  # Example expansion distance
