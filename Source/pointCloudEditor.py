import open3d as o3d


def open_point_cloud_editor(pcd: o3d.cpu.pybind.geometry.PointCloud) -> None:
    """A function to open a window that allows the point clouds to be cropped.
    The export of the cropped clouds are in PLY format.
    It has to be noted that when exporting the cutouts to PLY format, Open3D will round off the colour values.
    This results in somewhat distorted colors when converting them back to LAS format.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be edited.
    """
    print("\n")  # noqa: E303
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])
