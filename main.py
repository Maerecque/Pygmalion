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
    print("Be aware that files that are created with this application cannot be used again with this application.")
    print("NOTE: This application is not suited for very large files, try to use files that are 350MB or smaller. \n Unless you are running on a high-end computer.")  # noqa: E501
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        pcd = grid_subsampling(pcd, 0.01)
        pcd_stat = remove_noise_statistical(pcd, True)
        pointcloud_dbscan(
            pcd_stat,
            eps=0.5,
            min_samples=5,
            keep_only_labels=False,
            keep_no_labels=True
        )
        open_point_cloud_editor(pcd)

    convert_ply_to_las(file_name)
