import cv2
import cv2.aruco as aruco
import sys
import os

# Set output directory
output_dir = "aruco_markers"
os.makedirs(output_dir, exist_ok=True)

# Get number of markers from CLI or default to 5
num_markers = int(sys.argv[1]) if len(sys.argv) > 1 else 5

# Choose a predefined dictionary
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# Marker size in pixels
marker_size = 200

print(f"🖨️ Generating {num_markers} ArUco marker(s)...")

for marker_id in range(num_markers):
    marker_img = aruco.drawMarker(aruco_dict, marker_id, marker_size)
    filename = os.path.join(output_dir, f"aruco_id_{marker_id}.png")
    cv2.imwrite(filename, marker_img)
    print(f"✅ Saved: {filename}")

print("\n✅ Done! All markers are in the 'aruco_markers' folder.")
