import numpy as np


def normalize_array(input_array: np.ndarray, is_colour: bool = False) -> np.ndarray:
    """A function made to normalize a NumPy ndarray.

    Args:
        input_array (numpy.ndarray): A NumPy ndarray to normalize.

        is_colour (bool, optional): A boolean value to divide the ndarray to the LAS colour standards to 0.
            Defaults to False.

    Raises:
        TypeError: If the input_array is not of the NumPy ndarray type, this error will be raised.

    Returns:
       numpy.ndarray: The normalized NumPy ndarray.
    """
    try:
        if type(input_array) is not np.ndarray: raise TypeError

        if is_colour is True: normalized_array = input_array / 65535
        else: normalized_array = (input_array - np.min(input_array)) / (np.max(input_array) - np.min(input_array))

        return normalized_array
    except TypeError:
        print("Given value is not the correct type; not a NumPy ndarray.")
