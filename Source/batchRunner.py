import os
from typing import Union
import itertools

from dbscanPointCloud import pointcloud_dbscan
from fileHandler import readout_LAS_file
from pointCloudAltering import (
    grid_subsampling,
    remove_noise_radius,
    remove_noise_statistical)
from pointCloudEditor import open_point_cloud_editor


class emptyPointCloudError(Exception): pass


def combination_maker(lst: list) -> list[tuple]:
    """A function to make all possible combination for the batch runner when multiple options are given.

    Args:
        lst (list): list of items to be combined.

    Returns:
        list: A list of tuples with every combination of the list that has been given.
    """
    for index, item in enumerate(lst):
        if type(item) != list:
            lst[index] = [item]
    output_lst = list(itertools.product(*lst))
    return output_lst


def print_combination(
    method: str,
    item: str,
    voxel_size: int,
    combination: tuple,
    amount_of_points: int
):
    """Literally a function to just print all of the parameters used in a scan.

    Args:
        method (str): Method used for noise removal.
        item (str): Path of the file that was used.
        voxel_size (int): Voxel size used during normalization.
        combination (tuple): Combination of the parameters used during scan.
        amount_of_points (int): Amount of points that were in the point cloud.
    """
    print("\n")
    print(f"File: {item}")
    print(f"Doing {method}")
    print("The following settings will be ran:")
    print(f"Voxel size:{voxel_size}")
    print(f"NB_points:{combination[0]}")
    print(f"Radius:{combination[1]}")
    print(f"EPS:{combination[2]}")
    print(f"Min_sample:{combination[3]}")
    print(f"Metric:{combination[4]}")
    print(f"Algorithm:{combination[5]}")
    print(f"Leaf size:{combination[6]}")
    print(f"Amount of points: {amount_of_points}")


def batch_running(
    input_list: list,
    do_radius: bool = False,
    do_statistical: bool = True,
    voxel_size: float = 0.05,
    radius_nb_points: Union[int, list[int]] = 10,
    radius_radius: Union[float, list[float]] = 0.1,
    statistical_nb_neighbors: Union[int, list[int]] = 20,
    statistical_std_ratio: Union[float, list[float]] = 2.0,
    visualize_noise: bool = False,
    dbscan_eps: Union[float, list[float]] = 0.1,
    dbscan_min_sample: Union[int, list[int]] = 20,
    dbscan_metric: Union[str, list[str]] = 'euclidean',
    dbscan_algorithm: Union[str, list[str]] = 'auto',
    dbscan_leaf_size: Union[int, list[int]] = 30,
    dbscan_keep_only_labels: bool = True,
    dbscan_keep_no_labels: bool = False,
    dbscan_visualize_all: bool = False,
    dbscan_visualize_end_result: bool = False
):
    """A function to run the dbscan in batches, to speed up the process of unit testing

    Args:
        input_list (list): List to be used as input, contains path locations of files to be scanned.

        do_radius (bool, optional): Whether to do radius noise removal.
            Defaults to False.

        do_statistical (bool, optional): Whether to do statistical noise removal.
            Defaults to True.

        voxel_size (float, optional): Distance between points that is allowed.
            Defaults to 0.05.

        radius_nb_points (int, optional): nb_points hyperparameter for the radius noise remover function.
            Defaults to 10.

        radius_radius (float or list[float], optional): radius hyperparameter for the radius noise remover function.
            Defaults to 0.1.

        statistical_nb_neighbors (int, optional): nb_neighbors hyperparameter for the statistical noise remover function.
            Defaults to 20.

        statistical_std_ratio (float or list[float], optional): std_ratio hyperparameter for the statistical noise remover function.
            Defaults to 2.0.

        visualize_noise (bool, optional): Wether to visualize the noise that will be removed from the point clouds.
            Defaults to False.

        dbscan_eps (float or list[float], optional): The maximum distance between two samples for one to be considered as in the neighborhood of
            the other. This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
            Defaults to 0.1.

        dbscan_min_sample (int or list[int], optional): The number of samples (or total weight) in a neighborhood for a point to be considered as
            a core point. This includes the point itself.
            Defaults to 20.

        dbscan_metric (str or list[str], optional): The metric to use when calculating distance between instances in a feature array.
            It must be one of the options allowed by :func:`sklearn.metrics.pairwise_distances` for its metric parameter.
            The following metrics can be used: ['chebyshev','cityblock','euclidean','hamming','nan_euclidean','sqeuclidean'].
            Defaults to 'euclidean'.

        dbscan_algorithm (str or list[str], optional): The algorithm used by NearestNeighbors module to compute pointwise distances
            and find nearest neighbors. See NearestNeighbors module documentation for details.
            The following metrics can be used: ['auto', 'ball_tree', 'kd_tree', 'brute'].
            Defaults to 'auto'.

        dbscan_leaf_size (int or list[int], optional): Leaf size passed to *BallTree* or *cKDTree*.
            This can affect the speed of the construction and query, as well as the memory required to store the tree.
            The optimal value depends on the nature of the problem.
            Defaults to 30.

        dbscan_keep_only_labels (bool, optional): Whether to keep only the labels from the dbscan.
            Defaults to True.

        dbscan_keep_no_labels (bool, optional): Whether to keep none of the labels from the dbscan.
            Defaults to False.

        dbscan_visualize_all (bool, optional): Whether to visualize all labels.
            Defaults to False.

        dbscan_visualize_end_result (bool, optional): Whether to visualize end result with original pointcloud and labels.
            Defaults to False.

    Raises:
        emptyPointCloudError: This error will be raised if the given point cloud is empty after noise removal.
    """
    print("Batch runner starting.")
    print(f"{len(input_list)} files will be processed in this batch.")
    for item in input_list:
        pcd = readout_LAS_file(item)
        pcd = grid_subsampling(pcd, voxel_size)
        if do_radius:
            try:
                parameter_list = [
                    radius_nb_points,
                    radius_radius,
                    dbscan_eps,
                    dbscan_min_sample,
                    dbscan_metric,
                    dbscan_algorithm,
                    dbscan_leaf_size
                ]
                combined_parameter_list = combination_maker(parameter_list)
                for combination in combined_parameter_list:
                    print_combination("Radius", item, voxel_size, combination, len(pcd.points))

                    pcd_radius = remove_noise_radius(
                        pcd,
                        visualize_noise,
                        nb_points=combination[0],
                        radius=combination[1]
                    )

                    # Check if the point cloud is empty after the noise has been removed.
                    if len(pcd_radius.points) == 0:
                        raise emptyPointCloudError

                    else:
                        pointcloud_dbscan(
                            pcd_radius,
                            eps=combination[2],
                            min_samples=combination[3],
                            keep_only_labels=dbscan_keep_only_labels,
                            keep_no_labels=dbscan_keep_no_labels,
                            visualize_all=dbscan_visualize_all,
                            metric=combination[4],
                            algorithm=combination[5],
                            leaf_size=combination[6],
                        )
                        pcd_radius = None
                        print("\n")

            except emptyPointCloudError:
                print("After the removal of outliers in the point cloud, nothing was left, therefore no DBScan was performed.")

        if do_statistical:
            try:
                parameter_list = [
                    statistical_nb_neighbors,
                    statistical_std_ratio,
                    dbscan_eps,
                    dbscan_min_sample,
                    dbscan_metric,
                    dbscan_algorithm,
                    dbscan_leaf_size
                ]
                combined_parameter_list = combination_maker(parameter_list)
                for combination in combined_parameter_list:
                    print_combination("Statistical", item, voxel_size, combination, len(pcd.points))

                    pcd_statistical = remove_noise_statistical(
                        pcd,
                        visualize_noise,
                        nb_neighbors=combination[0],
                        std_ratio=combination[1]
                    )

                    # Check if the point cloud is empty after the noise has been removed.
                    if len(pcd_statistical.points) == 0:
                        raise emptyPointCloudError

                    else:
                        pcd_cluster = pointcloud_dbscan(
                            pcd_statistical,
                            eps=combination[2],
                            min_samples=combination[3],
                            keep_only_labels=dbscan_keep_only_labels,
                            keep_no_labels=dbscan_keep_no_labels,
                            visualize_all=dbscan_visualize_all,
                            metric=combination[4],
                            algorithm=combination[5],
                            leaf_size=combination[6]
                        )

                        if dbscan_visualize_end_result:
                            pcd_combined = pcd_cluster + remove_noise_statistical(
                                pcd,
                                visualize_noise,
                                nb_neighbors=combination[0],
                                std_ratio=combination[1]
                            )
                            open_point_cloud_editor(pcd_combined, False)

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
        dbscan_metric=[
            'chebyshev',
            'cityblock',
            'euclidean',
            'hamming',
            'nan_euclidean',
            'sqeuclidean',
        ],
        dbscan_keep_only_labels=False,
        dbscan_visualize_all=False,
        dbscan_algorithm=[
            'auto',
            'ball_tree',
            'kd_tree',
            'brute'           # Very powerful method, crashes most of the time and takes very longs
        ],
    )
