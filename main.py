import sys
import os

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

# from Source.dbscanPointCloud import pointcloud_dbscan
from Source.fileHandler import (
    # convert_ply_to_las,
    get_file_path,
    readout_LAS_file
)
from Source.pointCloudAltering import (
    grid_subsampling,
    remove_noise_statistical,
    # combine_point_cloud
)
from Source.pointCloudEditor import open_point_cloud_editor
from Source.shapeUtils import (
    repair_point_cloud_module,
    transform_mesh_to_pcd
)
import Source.gridRansacModule as zzr


if __name__ == "__main__":
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    print("\n Starting PointCloud module")

    # # Print a nice line over the width of the terminal
    term_size = os.get_terminal_size()
    print(u'\u2500' * term_size.columns)

    if pcd is not None:
        # Downsample the point cloud.
        pcd = grid_subsampling(pcd, 0.025)
        open_point_cloud_editor(pcd)

        # Remove noise from the point cloud
        pcd_stat = remove_noise_statistical(pcd, True)

        # Divide the pointcloud into a 3d grid
        grid = zzr.divide_pointcloud_into_grid(pcd_stat, 1, 0)

        plane_pointcloud = zzr.walk_through_grid(pcd_stat, grid, 250, 500)

        open_point_cloud_editor(plane_pointcloud, False)

        # Repair the point cloud with the planes found.
        # With these parameters, just try higher NN and depth and lower quantile value (and maybe scale)
        mesh = repair_point_cloud_module(plane_pointcloud, visualize=True, kdtree_max_nn=100, depth=13, quantile_value=0.01, scale=2.2)

        # Transform the mesh back to a point cloud.
        transformed_pcd = transform_mesh_to_pcd(mesh, plane_pointcloud)

        open_point_cloud_editor(transformed_pcd, False)




    # # ____________________________ PLY MODULE ____________________________ # noqa: E303
    # # When the point cloud alterations are done, the point cloud is saved as a PLY file or no LAS file is given, this part will start.
    # print("\n Starting PLY module")

    # # Print a nice line over the width of the terminal
    # print(u'\u2500' * term_size.columns)

    # convert_ply_to_las(file_name)
