import laspy
import numpy as np
import open3d as o3d

dataset_desk = ".//Werfkelderscans/Geomaat/Handscanner/121601-GeoSLAM-DeskWithWorker.las"

dataset_desk = "Productcode/Werfkelderscans/Geomaat/Handscanner/121601-GeoSLAM-DeskWithWorker.laz"


def readout_LAS_file(filename):
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        las = laspy.read(dataset_desk)

        point_data = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))

        geom = o3d.geometry.PointCloud()
        geom.points = o3d.utility.Vector3dVector(point_data)
        print("A " + str(geom)[:-1] + " was extracted from the given LAS/LAZ file.")
        return geom
    except FileNotFoundError:
        print("Could not find a file on the given PATH, please check if the PATH exists.")
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and/or if it is a LAS/LAZ file.")  # noqa: E501
    except Exception as e:
        print("An unforeseen error occurred. See below for details.")
        print(type(e))
        print(e)


pcd = readout_LAS_file(dataset_desk)
o3d.visualization.draw_geometries([pcd])
