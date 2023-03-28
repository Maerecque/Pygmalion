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
    remove_noise_statistical)
from Source.pointCloudEditor import open_point_cloud_editor


if __name__ == "__main__":
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        pcd = grid_subsampling(pcd, 0.05)
        pcd_stat = remove_noise_statistical(pcd, True)
        pointcloud_dbscan(
            pcd_stat,
            eps=0.1,
            min_samples=20,
            keep_only_labels=False,
            visualize_all=True
        )
        open_point_cloud_editor(pcd)

    convert_ply_to_las(file_name)
