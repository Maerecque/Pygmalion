import open3d as o3d
import sys
import os
import numpy as np


sys.path.insert(0, os.path.realpath(os.path.dirname(__file__)) + '\\Source')

from Source.fileHandler import get_file_path, readout_LAS_file
from Source.heightMapModule import transform_pointcloud_to_height_map


def load_and_preprocess_point_cloud() -> o3d.geometry.PointCloud:
    """
    Loads a point cloud file selected by the user and applies voxel downsampling and normal estimation.
    Returns:
        o3d.geometry.PointCloud: The processed point cloud.
    """
    pcd = readout_LAS_file(get_file_path("Select a point cloud file to process", "LAS files (*.las *.laz)"))
    print("Point cloud loaded and preprocessed. Now starting surface reconstruction...")
    return pcd


def main():
    pcd = load_and_preprocess_point_cloud()

    pcd = pcd.voxel_down_sample(voxel_size=0.05)

    new_pcd_tuple = transform_pointcloud_to_height_map(
        pcd,
        grid_size=150,
        visualize_map=True,
        debugging_logs=False
    )

    # put tuple into a single point cloud
    new_pcd = o3d.geometry.PointCloud()
    new_pcd.points = o3d.utility.Vector3dVector(
        np.vstack([
            np.asarray(new_pcd_tuple[0].points),
            np.asarray(new_pcd_tuple[1].points),
            np.asarray(new_pcd_tuple[2].points)]
        )
    )
    new_pcd.colors = o3d.utility.Vector3dVector(
        np.vstack([
            np.asarray(new_pcd_tuple[0].colors),
            np.asarray(new_pcd_tuple[1].colors),
            np.asarray(new_pcd_tuple[2].colors)]
        )
    )


if __name__ == "__main__":
    main()
