import cv2
import numpy as np
import glob

def calibrate_camera_from_images(image_folder='calibration_images', grid_size=(9, 6), square_size=1.0):
    # Termination criteria for cornerSubPix
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    objp = np.zeros((grid_size[0] * grid_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:grid_size[0], 0:grid_size[1]].T.reshape(-1, 2)
    objp *= square_size

    objpoints = []  # 3D points
    imgpoints = []  # 2D points

    images = glob.glob(f"{image_folder}/*.png")

    if not images:
        print("No images found.")
        return

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Output result to console
    print("\nCamera matrix:\n", mtx)
    print("Distortion coefficients:\n", dist)

    # Save to a .txt file
    with open("calibration_output.txt", "w") as f:
        f.write("camera_matrix = np.array(" + np.array2string(mtx, separator=', ') + ")\n")
        f.write("dist_coeffs = np.array(" + np.array2string(dist, separator=', ') + ")\n")

    print("\n✅ Calibration data saved to calibration_output.txt")

# Run the function
if __name__ == "__main__":
    calibrate_camera_from_images(
        image_folder="calibration_images",
        grid_size=(9, 6),  # inner corners
        # square_size=1.0    # use real-world size if known, else 1.0 is fine
        square_size = 18.0  # mm

    )
