"""
How to implement in code: 

from load_dictionary import load_custom_aruco_dict

custom_dict = load_custom_aruco_dict("aruco_markers/custom_dict.npy")

# Then you can use:
# aruco.detectMarkers(frame, custom_dict)

"""

import numpy as np
import cv2.aruco as aruco
import os

def load_custom_aruco_dict(dict_path="aruco_markers/custom_dict.npy"):
    """
    Loads a custom ArUco dictionary from a .npy file.

    Args:
        dict_path (str): Path to the saved .npy file containing bytesList.

    Returns:
        cv2.aruco_Dictionary: The reconstructed custom ArUco dictionary.
    """
    if not os.path.exists(dict_path):
        raise FileNotFoundError(f"Dictionary file not found: {dict_path}")

    loaded_bytes = np.load(dict_path)
    num_markers = loaded_bytes.shape[0]
    marker_bits = int(np.sqrt(loaded_bytes.shape[1] * 8))

    custom_dict = aruco.custom_dictionary(nMarkers=num_markers, markerSize=marker_bits)
    custom_dict.bytesList = loaded_bytes

    print(f"✅ Loaded custom ArUco dictionary: {num_markers} markers of size {marker_bits}x{marker_bits}")
    return custom_dict



