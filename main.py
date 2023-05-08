import sys
import os

# This line is needed so the scripts from the source folder are imported correctly without the need of an __init__ file.
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

from Source.dbscanPointCloud import pointcloud_dbscan
from Source.fileHandler import (
    convert_ply_to_las,
    get_file_path,
    readout_LAS_file)
from Source.pointCloudAltering import (  # noqa: F401
    grid_subsampling,
    remove_noise_radius,
    remove_noise_statistical,
    combine_point_cloud)
from Source.pointCloudEditor import open_point_cloud_editor
from Source.shape_utils import find_plane_module, repair_point_cloud_module


if __name__ == "__main__":
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    print("\n Starting DBScan module")

    # Print a nice line over the width of the terminal
    term_size = os.get_terminal_size()
    print(u'\u2500' * term_size.columns)

    if pcd is not None:
        pcd = grid_subsampling(pcd, 0.05)
        pcd_stat = remove_noise_statistical(pcd, True)
        pcd_cluster = pointcloud_dbscan(
            pcd_stat,
            eps=0.1,
            min_samples=20,
            keep_no_labels=True,
            visualize_only_labels=True,
            metric="cityblock"
        )

        pcd_combined = combine_point_cloud(remove_noise_statistical(pcd, False, print_removal_amount=False), pcd_cluster)

        open_point_cloud_editor(pcd_combined)

        # Yeees, my genius is sometimes frightening.
        plane_from_pcd = find_plane_module(pcd)
        open_point_cloud_editor(plane_from_pcd)

        # With these parameters, just try higher NN and depth and lower quantile value (and maybe scale)
        repair_point_cloud_module(pcd, visualize=True, kdtree_max_nn=100, depth=13, quantile_value=0.01, scale=2.2)


    # ____________________________ PLY MODULE ____________________________ # noqa: E303
    # When the point cloud alterations are done, the point cloud is saved as a PLY file or no LAS file is given, this part will start.
    print("\n")
    print("Starting PLY module")

    # Print a nice line over the width of the terminal
    print(u'\u2500' * term_size.columns)

    convert_ply_to_las(file_name)
