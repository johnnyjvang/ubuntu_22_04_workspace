import cv2
import os

def capture_chessboard_images_with_spacebar(
    output_folder="calibration_images",
    grid_size=(9, 6),
    start_index=1
):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam.")
        return

    img_count = start_index
    print("Press SPACE to save frame with detected chessboard. Press Q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, grid_size, None)

        display_frame = frame.copy()

        if found:
            cv2.drawChessboardCorners(display_frame, grid_size, corners, found)
            print("✅ Corners detected.")
        else:
            print("❌ Corners not detected.")

        cv2.imshow("Chessboard Capture", display_frame)
        key = cv2.waitKey(1)

        if key == ord(' '):  # If spacebar is pressed
            if found:
                filename = f"{output_folder}/image_{img_count:02d}.png"
                cv2.imwrite(filename, display_frame)  # Save the frame with corners drawn
                print(f"✅ Saved {filename}")
                img_count += 1
            else:
                print("⚠️ Chessboard not found — not saved.")

        elif key == ord('q'):  # If 'q' is pressed, exit
            print("👋 Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

# Run the function
if __name__ == "__main__":
    capture_chessboard_images_with_spacebar(
        output_folder="calibration_images",
        grid_size=(9, 6),
        start_index=1
    )
