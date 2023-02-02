import laspy
import numpy as np
import open3d as o3d

dataset_desk = ".//Werfkelderscans/Geomaat/Handscanner/121601-GeoSLAM-DeskWithWorker.las"


def normalize_array(inputArray):
    """A function made to normalize a NumPy ndarray.

    Args:
        inputArray (numpy.ndarray): A NumPy ndarray to normalize.

    Returns:
       numpy.ndarray: The normalized NumPy ndarray.
    """
    try:
        if type(inputArray) is not np.ndarray: raise TypeError
        normalizedArray = (inputArray - np.min(inputArray)) / (np.max(inputArray) - np.min(inputArray))

        return normalizedArray
    except TypeError:
        print("Given value is not the correct type; not a NumPy ndarray.")


def readout_LAS_file(filename):
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Returns:
        open3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        las = laspy.read(filename)

        geom = o3d.geometry.PointCloud()

        # Create an Open3d model that contains the points from the LAS/LAZ file.
        pointData = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
        geom.points = o3d.utility.Vector3dVector(pointData)

        # Assign the colours of the points to the Open3d model. Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly. # noqa: E501
        colourData = np.stack([normalize_array(las.red), normalize_array(las.green), normalize_array(las.blue)], axis=0).transpose((1, 0))  # noqa: E501
        geom.colors = o3d.utility.Vector3dVector(colourData)

        print("A " + str(geom)[:-1] + " was extracted from the given LAS/LAZ file.")
        return geom
    except FileNotFoundError:
        print("Could not find a file on the given PATH, please check if the PATH exists.")
        exit()
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and/or if it is a LAS/LAZ file.")  # noqa: E501
    except Exception as e:
        print("An unforeseen error occurred. See below for details.")
        print(type(e))
        print(e)


def remove_noise(inputPointCloud, showExample=False, showRemovedPoints=False):
    """A function to remove noise from a point cloud.
    !!! This function is still an experimental feature. !!!
    Right now remove_statistical_outlier is used, but radius_outlier_removal also exists and is probably worth looking into. 👀

    Args:
        inputPointCloud (open3d.cpu.pybind.geometry.PointCloud): A point cloud where the noise will be removed from.
        showExample (bool, optional): Boolean to show an example of the point cloud when the noise removal is done. Defaults to False.  # noqa; E501
        showRemovedPoints (bool, optional): Boolean to show an example with the removed points from the cloud marked in red. Defaults to False. # noqa; E501

    Returns:
        open3d.cpu.pybind.geometry.PointCloud): A cleaned up version of the point cloud.
    """
    cl, ind = inputPointCloud.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)

    if showExample: o3d.visualization.draw_geometries([cl])

    if showRemovedPoints:
        outlier_cloud = inputPointCloud.select_by_index(ind, invert=True)
        outlier_cloud.paint_uniform_color([1, 0, 0])
        o3d.visualization.draw_geometries([outlier_cloud, inputPointCloud])

    return cl


def reconstruct_surface(input_point_cloud):
    """A function to recreate the surfaces of the given point cloud.
    !!! This function is still very unstable and yet doesn't show any meaningful or useful results !!!

    Args:
        input_point_cloud (open3d.cpu.pybind.geometry.PointCloud): Point cloud to have surfaces reconstructed

    Returns:
       open3d.cpu.pybind.geometry.TriangleMesh: A TriangleMesh with the recreated surfaces of the given point cloud.
    """
    alpha = 75
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()

    return mesh


pcd = readout_LAS_file(dataset_desk)
o3d.visualization.draw_geometries([pcd])
