import cv2
import cv2.aruco as aruco
import numpy as np
from load_dictionary import load_custom_aruco_dict

# Load the custom dictionary
custom_dict = load_custom_aruco_dict("my_markers/custom_dict.npy")

# Load camera (default index 0)
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Could not open webcam.")
    exit()

# NOTE: Replace these with your actual calibration values
camera_matrix = np.array([[800, 0, 320],
                          [0, 800, 240],
                          [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros((5, 1))  # assuming no lens distortion

# Marker physical size (in meters, adjust to your real size)
marker_length = 0.05

# Rotation tracking variables
previous_theta = None
clockwise_total = 0
counterclockwise_total = 0

print("🎯 Tracking Marker ID: 0 (press 'q' to quit)")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, custom_dict)

    if ids is not None:
        for i, marker_id in enumerate(ids.flatten()):
            if marker_id == 0:
                # Estimate pose
                rvec, tvec, _ = aruco.estimatePoseSingleMarkers(
                    corners[i], marker_length, camera_matrix, dist_coeffs
                )

                # Draw marker and axis
                aruco.drawDetectedMarkers(frame, [corners[i]], ids[i])
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec, tvec, 0.03)

                # Extract position (x, y, z)
                x, y, z = tvec[0][0]

                # Extract rotation angle (theta) from rvec using Rodrigues
                rotation_matrix, _ = cv2.Rodrigues(rvec[0])
                theta = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
                theta_deg = np.degrees(theta)

                # Track rotation direction
                if previous_theta is not None:
                    delta_theta = theta - previous_theta
                    if delta_theta > 0:
                        clockwise_total += abs(np.degrees(delta_theta))
                    elif delta_theta < 0:
                        counterclockwise_total += abs(np.degrees(delta_theta))

                previous_theta = theta

                # Display position and theta
                text = f"x: {x:.2f} m, y: {y:.2f} m, z: {z:.2f} m, θ: {theta_deg:.1f}°"
                cv2.putText(frame, text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Show running clockwise/counterclockwise in degrees
    rotation_text = (f"Clockwise Total: {clockwise_total:.2f} degrees | "
                     f"Counterclockwise Total: {counterclockwise_total:.2f} degrees")
    cv2.putText(frame, rotation_text, (10, frame.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # Show live feed
    cv2.imshow("Tracking Marker ID 0", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == 32:  # Spacebar pressed
        print(f"🔁 Totals before reset — Clockwise: {clockwise_total:.2f}°, Counterclockwise: {counterclockwise_total:.2f}°")
        clockwise_total = 0.0
        counterclockwise_total = 0.0
        print("🔁 Totals have been reset.")

cap.release()
cv2.destroyAllWindows()
