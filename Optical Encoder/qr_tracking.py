import cv2
from pyzbar.pyzbar import decode
import time
import sys

# Get camera index from command-line or default to 0
camera_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

# Try to open the specified camera index
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

    # Flip frame for easier visual consistency
    frame = cv2.flip(frame, 1)

    # Decode QR codes in the current frame
    qr_codes = decode(frame)

    # Process each QR code detected
    for qr in qr_codes:
        # Get the QR code's position (its center)
        points = qr.polygon
        if len(points) == 4:
            x = int((points[0][0] + points[2][0]) / 2)
            y = int((points[0][1] + points[2][1]) / 2)

            # Draw rectangle around the QR code
            cv2.polylines(frame, [np.array(points)], True, (0, 255, 0), 2)

            # Optionally, display the decoded QR data
            qr_data = qr.data.decode('utf-8')
            cv2.putText(frame, qr_data, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Count revolutions based on vertical movement of the QR code's center
            if previous_position is not None:
                if previous_position > frame.shape[0] // 2 and y <= frame.shape[0] // 2:
                    if not crossed:
                        revolution_count += 1
                        print(f"Revolution #{revolution_count}")
                        crossed = True
                elif y > frame.shape[0] // 2:
                    crossed = False

            previous_position = y

    # Show the live frame with QR code tracking
    cv2.imshow("QR Code Tracking", frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

print(f"\n🟢 Total revolutions detected: {revolution_count}")
cap.release()
cv2.destroyAllWindows()
