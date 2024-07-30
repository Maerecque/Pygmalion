import os
import sys
import open3d as o3d

import numpy as np
import pyvista as pv

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

# from Source.dbscanPointCloud import pointcloud_dbscan
import Source.gridRansacModule as grm
from Source.fileHandler import (
    # convert_ply_to_las,
    get_file_path,
    get_save_file_path,
    readout_LAS_file
)
from Source.pointCloudAltering import (
    # combine_point_cloud,
    grid_subsampling,
    remove_noise_statistical
)
from Source.pointCloudEditor import open_point_cloud_editor
from Source.shapeUtils import repair_point_cloud_module, transform_mesh_to_pcd
from Source.meshAlterer import (
    mesh_simple_downsample,
    transform_mesh_to_height_map,
    transform_pcd_to_mesh
)

if __name__ == "__main__":
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        print("\n Starting PointCloud module")

        # Print a nice line over the width of the terminal
        term_size = os.get_terminal_size()
        print(u'\u2500' * term_size.columns)

        # Downsample the point cloud.
        pcd = grid_subsampling(pcd, 0.025)

        # Remove noise from the point cloud
        pcd_stat = remove_noise_statistical(pcd, False)

        # Compute normals
        pcd_stat.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

        # Create a mesh from the point cloud
        stat_mesh = repair_point_cloud_module(
            pcd_stat, visualize=False,
            kdtree_max_nn=100,
            depth=13,
            quantile_value=0.01,
            scale=2.2
        )

        # NEW CODE #

        # Downsample and simplify the mesh
        simplified_mesh = mesh_simple_downsample(stat_mesh, pcd_stat, 0.01, False)

        # Transform the mesh into a height map
        floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud = transform_mesh_to_height_map(simplified_mesh, 100, False, debugging_logs=False)

        # Transform the point clouds into a mesh
        floor_plan_volume = transform_pcd_to_mesh(floor_plan_point_cloud, bool_3d_mesh=False, alpha=0.1, tollerance=0.000001, offset=1)
        ceiling_volume = transform_pcd_to_mesh(ceiling_point_cloud, bool_3d_mesh=True, alpha=0.2, tollerance=0.000001, offset=1, visualize_bool=False)
        wall_volume = transform_pcd_to_mesh(wall_point_cloud, bool_3d_mesh=True, alpha=0.2, tollerance=0.000001, offset=1, visualize_bool=True)
        exit()

        # Create a filename location for the height map in stl
        export_file_path = get_save_file_path(
            "STL files", ["*.stl"],
            (str(os.path.basename(file_name).split(".")[0]) + ".stl")
        )

        try:
            # Export the height map as STL
            pv.save_meshio(export_file_path, volume)

        # Except type error
        except TypeError:
            print("No file save location given.")
            exit()

        exit()


        hull_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(hull_point_cloud, alpha=100)

        # Create a mesh from the height map
        repaired_hull_mesh = repair_point_cloud_module(hull_point_cloud, visualize=True, kdtree_max_nn=100, depth=13, quantile_value=0.01, scale=2.2)

        # # Visualize the contour vertices
        # o3d.visualization.draw_geometries([height_map_mesh], mesh_show_back_face=True)

        # CODE WORKS UNTIL HERE #
        exit()

        # Drape the mesh downward
        downward_mesh = drape_mesh_downward(simplified_mesh, interpolator, kdtree, valid_points, height_map, x_grid, y_grid, min_height) # Does not work yet

        # Visualize the downward mesh
        o3d.visualization.draw_geometries([downward_mesh], mesh_show_back_face=True)

        # Export the mesh as STL
        export_file_path = get_save_file_path("STL files", ["*.stl"],(str(os.path.basename(file_name))+".stl"))
        o3d.io.write_triangle_mesh(export_file_path, smooth_mesh)

        # END NEW CODE #

        # # Divide the pointcloud into a 3d grid
        # grid = grm.divide_pointcloud_into_grid(pcd_stat, 1, 0)

        # plane_pointcloud = grm.walk_through_grid(pcd_stat, grid, 250, 500)

        # open_point_cloud_editor(plane_pointcloud, False)

        # # Repair the point cloud with the planes found.
        # # With these parameters, just try higher NN and depth and lower quantile value (and maybe scale)
        # mesh = repair_point_cloud_module(plane_pointcloud, visualize=True, kdtree_max_nn=100, depth=13, quantile_value=0.01, scale=2.2)

        # # Transform the mesh back to a point cloud.
        # transformed_pcd = transform_mesh_to_pcd(mesh, plane_pointcloud)

        # open_point_cloud_editor(transformed_pcd, False)




    # # ____________________________ PLY MODULE ____________________________ # noqa: E303
    # # When the point cloud alterations are done, the point cloud is saved as a PLY file or no LAS file is given, this part will start.
    # print("\n Starting PLY module")

    # # Print a nice line over the width of the terminal
    # print(u'\u2500' * term_size.columns)

    # convert_ply_to_las(file_name)
