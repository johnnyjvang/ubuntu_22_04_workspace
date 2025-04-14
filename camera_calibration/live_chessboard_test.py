import cv2

def live_chessboard_corner_view(grid_size=(9, 6)):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("❌ Could not open webcam.")
        return

    print("📷 Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, grid_size, None)

        if found:
            # Draw corners on the original frame
            cv2.drawChessboardCorners(frame, grid_size, corners, found)

        cv2.imshow("Live Chessboard Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    live_chessboard_corner_view(grid_size=(9, 6))
