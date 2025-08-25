import open3d as o3d


def open_point_cloud_editor(pcd: o3d.cpu.pybind.geometry.PointCloud, show_help: bool = True) -> None:
    """A function to open a window that allows the point clouds to be cropped.
    The export of the cropped clouds are in PLY format.
    It has to be noted that when exporting the cutouts to PLY format, Open3D will round off the colour values.
    This results in somewhat distorted colors when converting them back to LAS format.

    Args:
        pcd (o3d.cpu.pybind.geometry.PointCloud): The point cloud to be edited.

        show_help (bool): Whether to show help messages.
            Defaults to True.
    """
    # Check if the input point cloud is empty
    if not isinstance(pcd, list) and len(pcd.points) == 0:
        # If the point cloud is empty, show an error message and return
        raise ValueError("The point cloud is empty")

    if show_help:
        print("\nOpen3D Geometry Editor Controls:")
        print("  1) Press 'Y' twice to align geometry with negative y-axis")
        print("  2) Press 'K' to lock screen and switch to selection mode")
        print("  3) Drag for rectangle selection, or use Ctrl+Left Click for polygon selection")
        print("  4) Press 'C' to crop selected geometry")
        print("  5) Press 'S' to save the selected geometry (as PLY)")
        print("  6) Press 'F' to switch to freeview mode")
        print("Note: Exported PLY colors may be rounded and appear distorted when converting back to LAS.")

    if not isinstance(pcd, list):
        o3d.visualization.draw_geometries_with_editing([pcd], left=0, top=45)
    if isinstance(pcd, list):
        o3d.visualization.draw_geometries_with_editing(pcd, left=0, top=45)
