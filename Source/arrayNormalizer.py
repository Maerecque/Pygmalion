import numpy as np


def normalize_array(inputArray: np.ndarray, isColour: bool = False) -> np.ndarray:
    """A function made to normalize a NumPy ndarray.

    Args:
        inputArray (numpy.ndarray): A NumPy ndarray to normalize.

        isColour (bool, optional): A boolean value to divide the ndarray to the LAS colour standards to 0.
            Defaults to False.

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
