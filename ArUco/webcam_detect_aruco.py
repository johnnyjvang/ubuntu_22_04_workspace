# python3 webcam_detect_aruco.py

import cv2
import cv2.aruco as aruco
from load_dictionary import load_custom_aruco_dict  # Make sure load_dictionary.py is in the same folder or PYTHONPATH

# Load custom ArUco dictionary
custom_dict = load_custom_aruco_dict("my_markers/custom_dict.npy")

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Unable to open webcam.")
    exit()

print("✅ Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame.")
        break

    # Convert frame to grayscale for ArUco detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect markers
    corners, ids, _ = aruco.detectMarkers(gray, custom_dict)

    # Draw detected markers on the frame
    if ids is not None:
        aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            print(f"🧩 Detected Marker ID: {marker_id}")

    # Display the video feed
    cv2.imshow("ArUco Marker Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
