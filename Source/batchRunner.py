import os

from dbscanPointCloud import pointcloud_dbscan
from fileHandler import readout_LAS_file
from pointCloudAltering import (
    grid_subsampling,
    remove_noise_radius,
    remove_noise_statistical)


class emptyPointCloudError(Exception): pass


def batch_running(
    input_list: list,
    do_radius: bool = True,
    do_statistical: bool = True,
    voxel_size: float = 0.05,
    radius_nb_points: int = 10,
    radius_radius: float = 0.1,
    statistical_nb_neighbors: int = 20,
    statistical_std_ratio: float = 2.0,
    visualize_noise: bool = False,
    dbscan_eps: float = 0.1,
    dbscan_min_sample: int = 20,
    dbscan_keep_only_labels: bool = True,
    dbscan_keep_no_labels: bool = False,
    dbscan_visualize_all: bool = False
):
    """A function to run the dbscan in batches, to speed up the process of unit testing

    Args:
        input_list (list): List to be used as input, contains path locations of files to be scanned.
        do_radius (bool, optional): Whether to do radius noise removal. Defaults to True.
        do_statistical (bool, optional): Whether to do statistical noise removal. Defaults to True.
        voxel_size (float, optional): Distance between points that is allowed. Defaults to 0.05.
        radius_nb_points (int, optional): nb_points hyperparameter for the radius noise remover function. Defaults to 10.
        radius_radius (float, optional): radius hyperparameter for the radius noise remover function. Defaults to 0.1.
        statistical_nb_neighbors (int, optional): nb_neighbors hyperparameter for the statistical noise remover function. Defaults to 20.
        statistical_std_ratio (float, optional): std_ratio hyperparameter for the statistical noise remover function. Defaults to 2.0.
        visualize_noise (bool, optional): Wether to visualize the noise that will be removed from the point clouds. Defaults to False.
        dbscan_eps (float, optional): eps hyperparameter for the db scan. Defaults to 0.1.
        dbscan_min_sample (int, optional): min_sample hyperparameter for the db scan. Defaults to 20.
        dbscan_keep_only_labels (bool, optional): Whether to keep only the labels from the dbscan. Defaults to True.
        dbscan_keep_no_labels (bool, optional): Whether to keep none of the labels from the dbscan. Defaults to False.
        dbscan_visualize_all (bool, optional): Whether to visualize all labels. Defaults to False.
    """
    print("Batch runner starting.")
    print(f"{len(input_list)} files will be processed in this batch.")
    for item in input_list:
        print(item)
        pcd = readout_LAS_file(item)
        pcd = grid_subsampling(pcd, voxel_size)
        if do_radius:
            try:
                print("Doing radius")
                pcd_radius = remove_noise_radius(
                    pcd,
                    visualize_noise,
                    nb_points=radius_nb_points,
                    radius=radius_radius
                )

                # Check if the point cloud is empty after the noise has been removed.
                if len(pcd_radius.points) == 0:
                    raise emptyPointCloudError
                else:
                    pointcloud_dbscan(
                        pcd_radius,
                        eps=dbscan_eps,
                        min_samples=dbscan_min_sample,
                        keep_only_labels=dbscan_keep_only_labels,
                        keep_no_labels=dbscan_keep_no_labels,
                        visualize_all=dbscan_visualize_all
                    )
                    pcd_radius = None

            except emptyPointCloudError:
                print("After the removal of outliers in the point cloud, nothing was left, therefore no DBScan was performed.")

        if do_statistical:
            try:
                print("Doing statistical")
                pcd_statistical = remove_noise_statistical(
                    pcd,
                    visualize_noise,
                    nb_neighbors=statistical_nb_neighbors,
                    std_ratio=statistical_std_ratio
                )

                # Check if the point cloud is empty after the noise has been removed.
                if len(pcd_statistical.points) == 0:
                    raise emptyPointCloudError
                else:
                    pointcloud_dbscan(
                        pcd_statistical,
                        eps=dbscan_eps,
                        min_samples=dbscan_min_sample,
                        keep_only_labels=dbscan_keep_only_labels,
                        keep_no_labels=dbscan_keep_no_labels,
                        visualize_all=dbscan_visualize_all
                    )
                    pcd_statistical = None

            except emptyPointCloudError:
                print("After the removal of outliers in the point cloud, nothing was left, therefore no DBScan was performed.")

    exit()


if __name__ == "__main__":
    head_folder = os.path.join(os.path.realpath(os.path.dirname(__file__)), '..')

    file_list_hand_scans = [
        head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4 - room1.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4 - room2.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4 - room3.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4 - room4.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4 - room5.las",  # noqa: E501
        # head_folder + "/Werfkelderscans/Geomaat/Handscanner/GerritGeoSlam/121601-GeoSLAM-Gerrit-4.laz"  # noqa: E501
    ]

    file_list_static_scans = [
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room1.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room2.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room3.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room4.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room5.las",  # noqa: E501
        head_folder + "/Werfkelderscans/Geomaat/Statisch/121602-Kelder van Gerrit 1 - room6.las",  # noqa: E501
    ]

    batch_running(
        file_list_static_scans,
        do_radius=False,
        voxel_size=0.05,
        dbscan_min_sample=30,
        dbscan_eps=0.1,
        radius_nb_points=10,
        radius_radius=0.1,
        statistical_nb_neighbors=20,
        statistical_std_ratio=2.0,
        dbscan_keep_no_labels=False,
        dbscan_keep_only_labels=True,
        dbscan_visualize_all=False
    )
