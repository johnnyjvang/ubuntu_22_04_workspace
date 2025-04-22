import cv2
import numpy as np
import time
import sys

# Get camera index from command-line or default to 0
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Define the lower and upper HSV range for orange
lower_orange = np.array([10, 100, 100])
upper_orange = np.array([25, 255, 255])

# Try the specified camera index
cap = cv2.VideoCapture(camera_index)
if not cap.isOpened():
    print(f"❌ Camera at index {camera_index} could not be opened.")
    exit()

print(f"✅ Using camera at index {camera_index}")
print("Starting... Press 'q' to quit.")

# Revolution tracking variables
revolution_count = 0
previous_position = None
crossed = False

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame. Exiting.")
        break

    # Flip and blur for consistency
    frame = cv2.flip(frame, 1)
    blurred = cv2.GaussianBlur(frame, (5, 5), 0)

    # Convert to HSV
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    # Create mask for orange
    mask = cv2.inRange(hsv, lower_orange, upper_orange)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # Find the largest contour
        c = max(contours, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)

        if radius > 5:
            # Draw the circle
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 140, 255), 2)
            center_y = int(y)

            if previous_position is not None:
                if previous_position > frame.shape[0] // 2 and center_y <= frame.shape[0] // 2:
                    if not crossed:
                        revolution_count += 1
                        print(f"Revolution #{revolution_count}")
                        crossed = True
                elif center_y > frame.shape[0] // 2:
                    crossed = False

            previous_position = center_y

    # Display the frame
    cv2.imshow("Tracking Orange Marker", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n🟠 Total revolutions detected: {revolution_count}")
cap.release()
cv2.destroyAllWindows()
