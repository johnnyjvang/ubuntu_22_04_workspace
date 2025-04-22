import cv2
import sys

# Check for camera index from CLI
if len(sys.argv) < 2:
    print("Usage: python camera_check.py <camera_index>")
    sys.exit(1)

try:
    cam_index = int(sys.argv[1])
except ValueError:
    print("❌ Please enter a valid integer for the camera index.")
    sys.exit(1)

# Try to open the specified camera
cap = cv2.VideoCapture(cam_index)
if not cap.isOpened():
    print(f"❌ Camera at index {cam_index} could not be opened.")
    sys.exit(1)

print(f"✅ Camera successfully opened at index {cam_index}")

# Show camera feed
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    cv2.imshow(f"Camera {cam_index}", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
