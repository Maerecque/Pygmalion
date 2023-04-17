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


if __name__ == "__main__":
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        pcd = grid_subsampling(pcd, 0.025)
        pcd_stat = remove_noise_statistical(pcd, True)
        pcd_cluster = pointcloud_dbscan(
            pcd_stat,
            eps=0.03,
            min_samples=10,
            keep_no_labels=True,
            visualize_only_labels=True,
            metric="chebyshev"
        )

        pcd_combined = combine_point_cloud(remove_noise_statistical(pcd, False, print_removal_amount=False), pcd_cluster)
        open_point_cloud_editor(pcd_combined)

    convert_ply_to_las(file_name)
