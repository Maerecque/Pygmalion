import sys
import os

print(os.path.realpath(os.path.dirname(__file__)) + '\\Source')
sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

from Source.dbscanPointCloud import pointcloud_dbscan  
from Source.fileHandler import (  
    convert_ply_to_las,
    get_file_path,
    readout_LAS_file)
from Source.pointCloudAltering import (  
    grid_subsampling,
    remove_noise_radius,
    remove_noise_statistical)
from Source.pointCloudEditor import open_point_cloud_editor


if __name__ == "__main__":
    print("Be aware that files that are created with this application cannot be used again with this application.")
    print("NOTE: This application is not suited for very large files, try to use files that are 350MB or smaller.")
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        pcd = grid_subsampling(pcd)
        pcd = remove_noise_radius(pcd)
        pcd = remove_noise_statistical(pcd)
        pointcloud_dbscan(pcd, eps=0.2)
        open_point_cloud_editor(pcd)

    convert_ply_to_las(file_name)
