import os
import sys
import open3d as o3d

import numpy as np
import pyvista as pv
from vtkmodules.vtkFiltersModeling import vtkFillHolesFilter

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
    transform_pcd_to_mesh
)
from Source.heightMapModule import transform_mesh_to_height_map  # noqa: F401

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

        # NEW CODE #
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

        # Downsample and simplify the mesh
        simplified_mesh = mesh_simple_downsample(stat_mesh, pcd_stat, 0.01, False)

        # Transform the mesh into a height map
        floor_plan_point_cloud, ceiling_point_cloud, wall_point_cloud = transform_mesh_to_height_map(simplified_mesh, 100, False)

        # Transform the point clouds into a mesh
        floor_plan_volume = transform_pcd_to_mesh(floor_plan_point_cloud, bool_3d_mesh=False, alpha=0.1, tollerance=0.000001, offset=1)  # noqa: E501
        ceiling_volume = transform_pcd_to_mesh(ceiling_point_cloud, bool_3d_mesh=True, alpha=0.2, tollerance=0.000001, offset=1)
        wall_volume = transform_pcd_to_mesh(wall_point_cloud, bool_3d_mesh=True, alpha=0.225, tollerance=0.000001, offset=1)

        # Combine all the parts into one volume
        volume = floor_plan_volume + ceiling_volume + wall_volume

        # Visualize the volume
        pv.plot(volume)

        # TODO: Fix remaining holes in the mesh
        # Transform the unstructured grid into a polydata
        poly_data = volume.extract_geometry()
        filler = vtkFillHolesFilter()
        filler.SetInputData(poly_data)
        filler.Update()

        vtk_volume = filler.GetOutput()

        # Create a filename location for the height map in stl
        export_file_path = get_save_file_path(
            "STL files", ["*.stl"],
            (str(os.path.basename(file_name).split(".")[0]) + ".stl")
        )

        try:
            # transform the vtk polydata to a pyvista mesh
            output_volume = pv.wrap(vtk_volume)

            # Export the height map as STL
            pv.save_meshio(export_file_path, output_volume)

        # Except type error
        except TypeError:
            print("No file save location given.")
            exit()

        exit()
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
