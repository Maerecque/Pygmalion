import numpy as np
import open3d as o3d
import laspy
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from plyfile import PlyData
import os


def convert_ply_to_las(inputLasPath: str):
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
        plyFile = PlyData.read(toConvertToLas, False)

        newLasFileName = os.path.splitext(toConvertToLas)[0] + ".las"
        LasHeaderPointFormat = laspy.read(inputLasPath).header.point_format
        LasHeaderFileVersion = laspy.read(inputLasPath).header.version
        outfile = laspy.create(point_format=LasHeaderPointFormat, file_version=LasHeaderFileVersion)

        outfile.x = plyFile['vertex']['x']
        outfile.y = plyFile['vertex']['y']
        outfile.z = plyFile['vertex']['z']
        outfile.red = plyFile['vertex']['red']
        outfile.green = plyFile['vertex']['green']
        outfile.blue = plyFile['vertex']['blue']

    except Exception as e:
        print("Something wen't wrong during conversion, the PLY file will not be deleted.")
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


def get_file_path(description: str, fileformat: any) -> str:
    """A function to get the filepath of a selected file.

    Args:
        description (str): Description of the to be selected file format.
        fileformat (any): Either a string of one specified file format or a list of file formats. e.g. "*.txt" or ["*.txt", "*.docx"] # noqa: E501

    Returns:
        str: The filepath of the selected file.
    """
    Tk().withdraw()  # we don't want a full GUI, so keep the root window from appearing
    filename = askopenfilename(filetypes=[(description, fileformat)])  # show an "Open" dialog box and return the path to the selected file # noqa: E501
    return filename


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


def readout_LAS_file(filename: str) -> o3d.cpu.pybind.geometry.PointCloud:
    """A function to read a LAS/LAZ file and convert the contents into an Open3D format.
    This makes it possible to use the Open3D tools on the LAS/LAZ files.

    Args:
        filename (str): A PATH to a LAS/LAZ file to be converted.

    Returns:
        o3d.cpu.pybind.geometry.PointCloud: An Open3D point cloud containing the contents of the LAS/LAZ file.
    """
    try:
        las = laspy.read(filename)

        print(las.header)

        geom = o3d.geometry.PointCloud()

        # Create an Open3d model that contains the points from the LAS/LAZ file.
        pointData = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
        geom.points = o3d.utility.Vector3dVector(pointData)

        # Assign the colours of the points to the Open3d model. Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly. # noqa: E501
        colourData = np.stack([normalize_array(las.red, True), normalize_array(las.green, True), normalize_array(las.blue, True)], axis=0).transpose((1, 0))  # noqa: E501
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


def crop_geometry():
    """A function to crop shapes out of a point cloud and save them in a separate LAS file.
    """
    print("Demo for manual geometry cropping")
    file_name = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])
    pcd = readout_LAS_file(file_name)


    print("\n")  # noqa: E303
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    o3d.visualization.draw_geometries_with_editing([pcd])

    convert_ply_to_las(file_name)


if __name__ == "__main__":
    crop_geometry()
