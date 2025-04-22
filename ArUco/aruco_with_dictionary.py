import cv2
import cv2.aruco as aruco
import numpy as np
import argparse
import os

# ----------------------------
# Parse CLI Arguments
# ----------------------------
parser = argparse.ArgumentParser(description="Generate multiple ArUco markers and save them as PNGs.")
parser.add_argument("--num", type=int, required=True, help="Number of unique markers to generate.")
parser.add_argument("--size", type=int, default=200, help="Pixel size of each marker image (default: 200).")
parser.add_argument("--output", type=str, default="aruco_markers", help="Directory to save marker images.")
args = parser.parse_args()

# ----------------------------
# Create Output Directory
# ----------------------------
os.makedirs(args.output, exist_ok=True)

# ----------------------------
# Create Custom Dictionary
# ----------------------------
marker_size_bits = 4  # 4x4 bits per marker
custom_dict = aruco.custom_dictionary(nMarkers=args.num, markerSize=marker_size_bits)

# Save dictionary for later detection use
dict_path = os.path.join(args.output, "custom_dict.npy")
np.save(dict_path, custom_dict.bytesList)
print(f"✅ Saved custom dictionary with {args.num} markers to: {dict_path}")

# ----------------------------
# Generate & Save Marker Images
# ----------------------------
for marker_id in range(args.num):
    marker_img = aruco.drawMarker(custom_dict, marker_id, args.size)
    filename = os.path.join(args.output, f"marker_{marker_id}.png")
    cv2.imwrite(filename, marker_img)
    print(f"🖨️  Saved marker ID {marker_id} to: {filename}")

# python3 aruco_with_dictionary.py --num 5 --size 300 --output my_markers

