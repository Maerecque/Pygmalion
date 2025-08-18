from tkinter import Tk
from tkinter.filedialog import askopenfilename, asksaveasfilename

import laspy
import numpy as np
import open3d as o3d
from plyfile import PlyData
import os, sys  # noqa: E401
sys.path.insert(1, "/".join(os.path.realpath(__file__).split("/")[0:-2]))
from Source.arrayNormalizer import normalize_array


class FileFormatError(Exception): pass   # noqa: E701
class NoFileGivenError(Exception): pass  # noqa: E701
class NoSaveLocationGivenError(Exception): pass  # noqa: E701
class NoPointCloudGivenError(Exception): pass  # noqa: E701


def get_file_path(description: str, fileformat: any, print_output: bool = True) -> str:
    """A function to get the filepath of a selected file.

    Args:
        description (str): Description of the to be selected file format.

        fileformat (any): Either a string of one specified file format or a list of file formats. e.g.
            "*.txt" or ["*.txt", "*.docx"].

        print_output (bool): Whether to print the selected file path to the console.

    Returns:
        str: The filepath of the selected file.
    """
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    current_folder = os.path.realpath(os.path.dirname(__file__))
    root.iconbitmap(current_folder + "\\support_files\\logo.ico")
    # show an "Open" dialog box and return the path to the selected file
    filename = askopenfilename(
        filetypes=[(description, fileformat)],
        initialdir=os.path.join(current_folder, '..')
    )
    if filename:
        if print_output:
            print("The following file was selected: \n" + filename)
        return filename

    return


def get_save_file_path(description: str, fileformat: any, default_name: str) -> str:
    """A function to get the filepath of a selected file to save.

    Args:
        description (str): Description of the to be selected file format.

        fileformat (any): Either a string of one specified file format or a list of file formats. e.g.
            "*.txt" or ["*.txt", "*.docx"].

        default_name (str): The default name of the file that will be saved.

    Returns:
        str: The filepath of the selected file.
    """
    root = Tk()
    root.withdraw()  # we don't want a full GUI, so keep the root window from appearing
    current_folder = os.path.realpath(os.path.dirname(__file__))
    root.iconbitmap(current_folder + "\\support_files\\logo.ico")
    # show a "save" dialog box and return the path to the selected file
    filename = asksaveasfilename(
        filetypes=[(description, fileformat)],          # filetypes=[("Text files", "*.txt"), ("all files", "*.*")]
        initialdir=os.path.join(current_folder, '..'),  # initialdir=os.path.join(current_folder, '..')
        initialfile=default_name                        # initialfile="default_name.txt"
    )
    if filename:
        print("The following file was selected: \n" + filename)
        return filename

    return


def readout_LAS_file(filename: str, prnt_bool: bool = True) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Raises:
        FileNotFoundError: If a given path does not exist.

        NoFileGivenError: If no file is selected.

        FileFormatError: If the format of the given file is not supported.

        laspy.errors.LaspyException: If Laspy runs into an error.


    Returns:
        o3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        if not filename:
            raise NoFileGivenError

        las = laspy.read(filename)

        # check if LAS file is in the correct format
        if "<LasData(1.2, point fmt: <PointFormat(3," not in str(las):
            if len(las.points) < 10000000:
                raise FileFormatError

        geom = o3d.geometry.PointCloud()

        scales = las.header.scales
        offsets = las.header.offsets

        # stack coordinates directly along axis=1 (shape: (N,3))
        point_data = np.stack([las.X, las.Y, las.Z], axis=1)

        # apply scaling and offset with broadcasting
        point_data = point_data * scales + offsets

        geom.points = o3d.utility.Vector3dVector(point_data)

        # normalize colors once, stack axis=1
        colour_data = np.stack([
            normalize_array(las.red, True),
            normalize_array(las.green, True),
            normalize_array(las.blue, True)
        ], axis=1)

        geom.colors = o3d.utility.Vector3dVector(colour_data)

        if prnt_bool:
            print("A " + str(geom)[:-1] + " was extracted from the given LAS/LAZ file.")

        return geom

    except FileNotFoundError:
        print("Could not find a file on the given PATH,  please check if the PATH exists.")
        exit()
    except NoFileGivenError:
        print("No file was selected, script will not be stopped.")
        return
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and if it is a LAS/LAZ file.")
        exit()
    except FileFormatError:
        print("The chosen LAS/LAZ file is not in the correct format or correct version. This file will not be used.")
        exit()
    except Exception as e:
        print("An unforeseen error occurred. See below for details.")
        print(type(e))
        print(e)
        exit()


def convert_ply_to_las(input_las_path: str = None):
    """A function to convert a ply file to a LAS file, based on a given LAS input file.
    With this function the user is prompted to select a PLY file that will be converted to a LAS file.
    If the function runs into an error during the conversion, no new LAS file will be created and the PLY file will be kept.

    !!! NOTE: This function will delete the PLY file once it's converted to a LAS file, unless it runs into an error !!!

    Args:
        input_las_path (str): The path to the LAS file to be used as header template.

    Raises:
        NoFileGivenError: If no file is selected.
    """
    try:
        print("Select your created ply file to convert it to a LAS file.")
        ply_to_convert_to_las = get_file_path("PLY files", "*.ply")

        # if no PLY file is selected this if statement will be ran.
        if not ply_to_convert_to_las:
            raise NoFileGivenError

        ply_file = PlyData.read(ply_to_convert_to_las, False)
        new_las_file_name = os.path.splitext(ply_to_convert_to_las)[0] + ".las"

        # if no LAS file is selected a custom header will be created for the new LAS file that will be made out of the PLY file.
        if not input_las_path:
            custom_header = laspy.LasHeader(version="1.2", point_format=3)
            las_header_point_format = custom_header.point_format
            las_header_file_version = custom_header.version

        else:
            las_header_point_format = laspy.read(input_las_path).header.point_format
            las_header_file_version = laspy.read(input_las_path).header.version

        outfile = laspy.create(point_format=las_header_point_format, file_version=las_header_file_version)

        outfile.x = ply_file['vertex']['x']
        outfile.y = ply_file['vertex']['y']
        outfile.z = ply_file['vertex']['z']
        outfile.red = ply_file['vertex']['red'] * 257
        outfile.green = ply_file['vertex']['green'] * 257
        outfile.blue = ply_file['vertex']['blue'] * 257

    except NoFileGivenError:
        print("No file was selected, script will be stopped.")
        exit()
    except Exception as e:
        print("Something went wrong during conversion, the PLY file will not be deleted.")
        print(e)
    else:
        outfile.write(new_las_file_name)
        print("The PLY data was saved to " + new_las_file_name)
        os.remove(ply_to_convert_to_las)
        print("The PLY file was successfully deleted")
        try:
            os.remove(os.path.splitext(ply_to_convert_to_las)[0] + ".json")
            print("The accompanying JSON file was successfully deleted.")
        except Exception as e:  # noqa: F841
            print("No accompanying JSON file was found.")


def save_pcd_as_las(input_pcd: o3d.cpu.pybind.geometry.PointCloud):
    """A function to save an Open3D point cloud as a LAS file.

    Args:
        input_pcd (o3d.cpu.pybind.geometry.PointCloud): An Open3D point cloud to be saved as a LAS file.

    Raises:
        TypeError: If the input is not an Open3D point cloud.
        NoPointCloudGivenError: If the input point cloud is empty.
        NoSaveLocationGivenError: If no save location is selected.
    """
    try:
        if not isinstance(input_pcd, o3d.cpu.pybind.geometry.PointCloud):
            raise TypeError("The input must be an Open3D point cloud.")

        if len(input_pcd.points) == 0:
            raise NoPointCloudGivenError

    except NoPointCloudGivenError:
        print("The input point cloud is empty. Please provide a valid point cloud.")

    except TypeError as e:
        print(e)

    try:
        file_name = get_save_file_path("LAS files", "*.las", "default_name.las")
        if not file_name:
            raise NoFileGivenError

        # Check if the file name ends with .las, if not, add it
        if not file_name.lower().endswith('.las'):
            file_name += '.las'

        # Create a new las file
        las_file = laspy.create(point_format=3, file_version='1.2')

        # Get the point data from the Open3D point cloud
        point_data = np.asarray(input_pcd.points)

        # Get the colour data from the Open3D point cloud
        colour_data = np.asarray(input_pcd.colors)

        # Assign the point data to the las file
        las_file.x = point_data[:, 0]
        las_file.y = point_data[:, 1]
        las_file.z = point_data[:, 2]

        # Assign the colour data to the las file
        las_file.red = colour_data[:, 0] * 257
        las_file.green = colour_data[:, 1] * 257
        las_file.blue = colour_data[:, 2] * 257

        # Write the las file to the given file name
        las_file.write(file_name)
        print("The point cloud was saved as a LAS file.")
    except NoSaveLocationGivenError:
        print("No save location was selected.")
    except Exception as e:
        print("Something went wrong during the saving process.")
        print(e)
