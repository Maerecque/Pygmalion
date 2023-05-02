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
from Source.shape_utils import (  # noqa: F401
    find_alpha_shapes,
    find_outside_pointcloud,
    keep_points_in_view,
    detect_planar_patches)


if __name__ == "__main__":
    print("Starting DBScan module")
    term_size = os.get_terminal_size()
    print(u'\u2500' * term_size.columns)
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

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

    # When the point cloud alterations are done, the point cloud is saved as a PLY file or no LAS file is given, this part will start.
    print("\n")
    print("Starting PLY module")
    print(u'\u2500' * term_size.columns)

    convert_ply_to_las(file_name)
