import open3d as o3d
import os


def visualize_point_cloud(pcd_path):
    point_cloud = o3d.io.read_point_cloud(pcd_path)
    o3d.visualization.draw_geometries([point_cloud], left=0, top=45)


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python open3d_visualization.py <path_to_pcd>")
        sys.exit(1)

    # Set the verbosity level of Open3D to only print severe errors
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Error)

    pcd_path = sys.argv[1]
    visualize_point_cloud(pcd_path)

    # Close the temporary PCD file
    if pcd_path.startswith("temp_"):
        try:
            os.remove(pcd_path)
        except Exception as e:
            print("An error occurred while deleting the temporary PCD file:", str(e))
