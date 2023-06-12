import os
from tkinter import Tk
from tkinter.filedialog import askopenfilename

import laspy
import numpy as np
import open3d as o3d
from arrayNormalizer import normalize_array
from plyfile import PlyData


class FileFormatError(Exception): pass
class noFileGivenError(Exception): pass


def get_file_path(description: str, fileformat: any) -> str:
    """A function to get the filepath of a selected file.

    Args:
        description (str): Description of the to be selected file format.

        fileformat (any): Either a string of one specified file format or a list of file formats. e.g. "*.txt" or ["*.txt", "*.docx"].

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
        print("The following file was selected: \n" + filename)
        return filename

    return


def open_stage_file(filename: str,) -> o3d.cpu.pybind.geometry.PointCloud:
    current_folder = os.path.realpath(os.path.dirname(__file__))
    filename_new = os.path.join(current_folder, '..', filename)
    return filename_new


def readout_LAS_file(filename: str, print_info: bool = True) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

        print_info (bool, optional): A boolean to determine if the function should print information about the LAS/LAZ file.
            Defaults to True.

    Raises:
        FileNotFoundError: If a given path does not exist.

        noFileGivenError: If no file is selected.

        FileFormatError: If the format of the given file is not supported.

        laspy.errors.LaspyException: If Laspy runs into an error.


    Returns:
        o3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        if not filename:
            raise noFileGivenError

        las = laspy.read(filename)

        # check if LAS file is in the correct format
        if "<LasData(1.2, point fmt: <PointFormat(3," not in str(las):
            if len(las.points) < 10000000:
                raise FileFormatError

        geom = o3d.geometry.PointCloud()

        # Create an Open3d model that contains the points from the LAS/LAZ file.
        point_data = np.stack([
            las.X,
            las.Y,
            las.Z
        ], axis=0).transpose((1, 0))

        # With the line below the visualization will look "odd", but is needed for the export to PLY and turn back to the LAS format.
        geom.points = o3d.utility.Vector3dVector((point_data * las.header.scales) + las.header.offsets)
        # geom.points = o3d.utility.Vector3dVector(point_data)

        # Assign the colours of the points to the Open3d model.
        # Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly.
        colour_data = np.stack([
            normalize_array(las.red, True),
            normalize_array(las.green, True),
            normalize_array(las.blue, True)
        ], axis=0).transpose((1, 0))
        geom.colors = o3d.utility.Vector3dVector(colour_data)

        # # It seems like Open3d does not accept the use gps as a variable. Find a way to give this to the ply file.
        # gpsData = np.stack([las.gps_time], axis=0).transpose((1, 0))
        # geom.gps = o3d.utility.Vector2dVector(gpsData)

        if print_info:
            print("A " + str(geom)[:-1] + " was extracted from the given LAS/LAZ file.")
        return geom
    except FileNotFoundError:
        print("Could not find a file on the given PATH,  please check if the PATH exists.")
        exit()
    except noFileGivenError:
        print("No file was selected, script will not be stopped.")
        return
    except laspy.errors.LaspyException:
        print("The framework could not handle this file, please check if the file is not corrupted and/or if it is a LAS/LAZ file.")
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
        noFileGivenError: If no file is selected.
    """
    try:
        print("Select your created ply file to convert it to a LAS file.")
        ply_to_convert_to_las = get_file_path("PLY files", "*.ply")

        # if no PLY file is selected this if statement will be ran.
        if not ply_to_convert_to_las:
            raise noFileGivenError

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

    except noFileGivenError:
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
