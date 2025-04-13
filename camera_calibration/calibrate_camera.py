import cv2
import numpy as np
import glob

def calibrate_camera(image_folder="calibration_images", grid_size=(9, 6), square_size=1.0):
    objp = np.zeros((grid_size[1]*grid_size[0], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2) * square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    images = glob.glob(f'{image_folder}/*.png')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)
    return mtx, dist

# Example usage
# mtx, dist = calibrate_camera()
