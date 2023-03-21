import laspy
import numpy as np
import open3d as o3d
# Visualize the image using OpenCV
import cv2


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


dataset_desk = get_file_path("LAS and LAZ files", ["*.las", "*.laz"])

las = laspy.read(dataset_desk)

geom = o3d.geometry.PointCloud()

# Create an Open3d model that contains the points from the LAS/LAZ file.
pointData = np.stack([las.X, las.Y, las.Z], axis=0).transpose((1, 0))
geom.points = o3d.utility.Vector3dVector(pointData)

# Assign the colours of the points to the Open3d model. Open3d only takes in colour values between 0 and 1, so therefore the colour values will be normalized accordingly. # noqa: E501
colourData = np.stack([normalize_array(las.red, True), normalize_array(las.green, True), normalize_array(las.blue, True)], axis=0).transpose((1, 0))  # noqa: E501
geom.colors = o3d.utility.Vector3dVector(colourData)

pcd = geom

# Load the point cloud data
pcd = o3d.io.read_point_cloud("point_cloud.pcd")

vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(pcd)
vis.run()
vis.destroy_window()

# Project the point cloud data onto a 2D plane
image = o3d.geometry.image_from_point_cloud(pcd)

# Convert the image to a numpy array
image = np.asarray(image)


cv2.imshow("Image", image)
cv2.waitKey(0)  

