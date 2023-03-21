import matplotlib as plt
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


def pointcloud_dbscan(
    pcd: o3d.cpu.pybind.geometry.PointCloud,
    eps: float = 0.1,
    min_samples: int = 20,
    visualize_all: bool = False,
    keep_only_labels: bool = True,
    keep_no_labels: bool = False
) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to perform a DBScan on a point cloud.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): Point cloud that will be used as input for the DBScan.
        eps (float, optional): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
            This is not a maximum bound on the distances of points within a cluster.
            This is the most important DBSCAN parameter to choose appropriately for your data set and distance function. Defaults to 0.1.
        min_samples (int, optional): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.
            This includes the point itself. Defaults to 20.
        visualize_all (bool, optional): A boolean parameter to toggle visualization. Defaults to False.
        keep_only_labels (bool, optional): A boolean parameter to toggle keep only labels. Defaults to True.
        keep_no_labels (bool, optional): A boolean parameter to toggle keep only points with no labels. Defaults to False.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: Point cloud with DBScan performed on it.
    """
    # Convert points to a numpy array.
    xyz = np.asarray(pcd.points)

    # Perform DBSCAN clustering.
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(xyz)

    print(str(len(np.unique(labels)) - 1) + " label(s) were made with dbscan")

    # Create a color map for the clusters.
    maxLabel = labels.max()
    colors = plt.cm.jet(labels / (maxLabel if maxLabel > 0 else 1))

    # Set colors for each point in the point cloud.
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])

    if visualize_all:
        o3d.visualization.draw_geometries([pcd], left=0, top=45, window_name="DBScan result with everything in it.")
        return pcd

    if keep_only_labels:
        xyz_colors = np.asarray(pcd.points)
        rgb_colors = np.asarray(pcd.colors)
        points = np.column_stack((xyz_colors, rgb_colors))

        # Define color to remove
        color_to_remove = [0.0, 0.0, 0.5]

        # Create boolean mask for points with the color to remove
        color_mask = np.all(points[:, 3:6] == color_to_remove, axis=1)

        # Remove points that satisfy the mask
        filtered_points = points[~color_mask]

        # Convert numpy array back to point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points[:, :3])
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_points[:, 3:6])
        o3d.visualization.draw_geometries([filtered_pcd], left=0, top=45, window_name="DBScan result with only labels left")
        return filtered_pcd

    if keep_no_labels:
        xyz_colors = np.asarray(pcd.points)
        rgb_colors = np.asarray(pcd.colors)
        points = np.column_stack((xyz_colors, rgb_colors))

        # Define color to keep
        color_to_keep = [0.0, 0.0, 0.5]

        # Create boolean mask for points with the color to keep
        color_mask = np.all(points[:, 3:6] == color_to_keep, axis=1)

        # Remove points that satisfy the mask
        filtered_points = points[color_mask]

        # Convert numpy array back to point cloud
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points[:, :3])
        filtered_pcd.colors = o3d.utility.Vector3dVector(filtered_points[:, 3:6])
        o3d.visualization.draw_geometries([filtered_pcd], left=0, top=45, window_name="DBScan result with no labels left")
        return filtered_pcd
