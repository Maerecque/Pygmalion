import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import laspy
import numpy as np
import open3d as o3d
from plyfile import PlyData


class FileFormatError(Exception): pass
class noFileGivenError(Exception): pass


def get_file_path(description: str, fileformat: any) -> str:
    """A function to get the filepath of a selected file.

    Args:
        description (str): Description of the to be selected file format.
        fileformat (any): Either a string of one specified file format or a list of file formats. e.g. "*.txt" or ["*.txt", "*.docx"] # noqa: E501

    Returns:
        str: The filepath of the selected file.
    """
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    root.iconbitmap(os.path.realpath(os.path.dirname(__file__)) + "\\support_files\\logo.ico")

    filename = askopenfilename(filetypes=[(description, fileformat)])  # show an "Open" dialog box and return the path to the selected file # noqa: E501
    if filename:
        print("The following file was selected: \n" + filename)
        return filename

    return


def readout_LAS_file(filename: str) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        if not filename:
            raise noFileGivenError

        las = laspy.read(filename)

        # check if LAS file is in the correct format
        if "<LasData(1.2, point fmt: <PointFormat(3," not in str(las):
            raise FileFormatError

        geom = o3d.geometry.PointCloud()

        # Create an Open3d model that contains the points from the LAS/LAZ file.
        pointData = np.stack([
            las.X,
            las.Y,
            las.Z
        ], axis=0).transpose((1, 0))
        geom.points = o3d.utility.Vector3dVector((pointData * las.header.scales) + las.header.offsets)

        # Assign the colours of the points to the Open3d model. Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly. # noqa: E501
        colourData = np.stack([
            normalize_array(las.red, True),
            normalize_array(las.green, True),
            normalize_array(las.blue, True)
        ], axis=0).transpose((1, 0))
        geom.colors = o3d.utility.Vector3dVector(colourData)

        # # It seems like Open3d does not accept the use gps as a variable. Find a way to give this to the ply file
        # gpsData = np.stack([las.gps_time], axis=0).transpose((1, 0))
        # geom.gps = o3d.utility.Vector2dVector(gpsData)

        print("A " + str(geom)[:-1] + " was extracted from the given LAS/LAZ file.")
        return geom
    except FileNotFoundError:
        print("Could not find a file on the given PATH,  please check if the PATH exists.")
        exit()
    except noFileGivenError:
        print("No file was selected, script will not be stopped.")
        return
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and/or if it is a LAS/LAZ file.")  # noqa: E501
        exit()
    except FileFormatError:
        print("The chosen LAS/LAZ file is not in the correct format or correct version. This file will not be used.")
        exit()
    except Exception as e:
        print("An unforeseen error occurred. See below for details.")
        print(type(e))
        print(e)
        exit()


def normalize_array(inputArray: np.ndarray, isColour: bool = False) -> np.ndarray:
    """A function made to normalize a NumPy ndarray.

    Args:
        inputArray (numpy.ndarray): A NumPy ndarray to normalize.
        isColour (bool, optional): A boolean value to divide the ndarray to the LAS colour standards to 0. Defaults to False.  # noqa: E501

    Raises:
        TypeError: If the inputArray is not of the NumPy ndarray type, this error will be raised.

    Returns:
       numpy.ndarray: The normalized NumPy ndarray.
    """
    try:
        if type(inputArray) is not np.ndarray: raise TypeError

        if isColour is True: normalizedArray = inputArray / 65535
        else: normalizedArray = (inputArray - np.min(inputArray)) / (np.max(inputArray) - np.min(inputArray))

        return normalizedArray
    except TypeError:
        print("Given value is not the correct type; not a NumPy ndarray.")


def open_point_cloud_editor(pcd: o3d.cpu.pybind.geometry.PointCloud) -> None:
    """A function to open a window that allows the point clouds to be cropped. The export of the cropped clouds are in PLY format.  # noqa: E501
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


def convert_ply_to_las(inputLasPath: str = None):
    """A function to convert a ply file to a LAS file, based on a given LAS input file.
    With this function the user is prompted to select a PLY file that will be converted to a LAS file.
    If the function runs into an error during the conversion, no new LAS file will be created and the PLY file will be kept.  # noqa: E501

    !!! NOTE: This function will delete the PLY file once it's converted to a LAS file, unless it runs into an error !!!

    Args:
        inputLasPath (str): The path to the LAS file to be used as header template.
    """
    try:
        print("Select your created ply file to convert it to a LAS file.")
        toConvertToLas = get_file_path("PLY files", "*.ply")

        # if no PLY file is selected this if statement will be ran.
        if not toConvertToLas:
            raise noFileGivenError

        plyFile = PlyData.read(toConvertToLas, False)
        newLasFileName = os.path.splitext(toConvertToLas)[0] + ".las"

        # if no LAS file is selected a custom header will be created for the new LAS file that will be made out of the PLY file.  # noqa: E501
        if not inputLasPath:
            customHeader = laspy.LasHeader(version="1.2", point_format=3)
            LasHeaderPointFormat = customHeader.point_format
            LasHeaderFileVersion = customHeader.version

        else:
            LasHeaderPointFormat = laspy.read(inputLasPath).header.point_format
            LasHeaderFileVersion = laspy.read(inputLasPath).header.version

        outfile = laspy.create(point_format=LasHeaderPointFormat, file_version=LasHeaderFileVersion)

        outfile.x = plyFile['vertex']['x']
        outfile.y = plyFile['vertex']['y']
        outfile.z = plyFile['vertex']['z']
        outfile.red = plyFile['vertex']['red'] * 257
        outfile.green = plyFile['vertex']['green'] * 257
        outfile.blue = plyFile['vertex']['blue'] * 257

    except noFileGivenError:
        print("No file was selected, script will be stopped.")
        exit()
    except Exception as e:
        print("Something went wrong during conversion, the PLY file will not be deleted.")
        print(e)
    else:
        outfile.write(newLasFileName)
        print("The PLY data was saved to " + newLasFileName)
        os.remove(toConvertToLas)
        print("The PLY file was successfully deleted")
        try:
            os.remove(os.path.splitext(toConvertToLas)[0] + ".json")
            print("The accompanying JSON file was successfully deleted.")
        except Exception as e:  # noqa: F841
            print("No accompanying JSON file was found.")


def grid_subsampling(points: laspy.LasReader, voxel_size: int) -> list:
    """Define a function that takes as input an array of points, and a voxel size expressed in meters.
    It returns the sampled point cloud.

    Not sure if this function should be used, but the idea is very potential

    Args:
        points (laspy.LasReader): _description_
        voxel_size (int): _description_

    Returns:
        list: _description_
    """
    # nb_vox = np.ceil((np.max(points, axis=0) - np.min(points, axis=0)) / voxel_size)
    non_empty_voxel_keys, inverse, nb_pts_per_voxel = np.unique(((points - np.min(points, axis=0)) // voxel_size).astype(int), axis=0, return_inverse=True, return_counts=True)  # noqa: E501
    idx_pts_vox_sorted = np.argsort(inverse)
    voxel_grid = {}
    grid_barycenter, grid_candidate_center = [], []
    last_seen = 0

    for idx, vox in enumerate(non_empty_voxel_keys):
        voxel_grid[tuple(vox)] = points[idx_pts_vox_sorted[last_seen:last_seen + nb_pts_per_voxel[idx]]]
        grid_barycenter.append(np.mean(voxel_grid[tuple(vox)], axis=0))
        grid_candidate_center.append(voxel_grid[tuple(vox)][np.linalg.norm(voxel_grid[tuple(vox)] - np.mean(voxel_grid[tuple(vox)], axis=0), axis=1).argmin()])  # noqa: E501
        last_seen += nb_pts_per_voxel[idx]

    return grid_candidate_center


if __name__ == "__main__":
    print("Be aware that files that are created with this application cannot be used again with this application.")  # noqa: E501
    print("NOTE: This application is not suited for very large files, try to use files that are 350MB or smaller.")
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)

    if pcd is not None:
        open_point_cloud_editor(pcd)

    convert_ply_to_las(file_name)
