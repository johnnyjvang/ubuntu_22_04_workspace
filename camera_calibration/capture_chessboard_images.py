import cv2
import os

def capture_chessboard_images(output_folder="calibration_images", num_images=20, grid_size=(9, 6)):
    os.makedirs(output_folder, exist_ok=True)
    cap = cv2.VideoCapture(0)
    count = 0

    while count < num_images:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret_corners, _ = cv2.findChessboardCorners(gray, grid_size, None)

        if ret_corners:
            filename = f"{output_folder}/img_{count:02d}.png"
            cv2.imwrite(filename, frame)
            print(f"Saved {filename}")
            count += 1

        # Show preview
        cv2.imshow("Capture Chessboard", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Example usage
# capture_chessboard_images(num_images=20, grid_size=(9, 6))
