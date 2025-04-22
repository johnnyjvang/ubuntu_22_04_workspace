import cv2

print("Checking available camera indices...")

for i in range(5):  # Test camera indices 0–4
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"✅ Camera found at index {i}")
        cap.release()
    else:
        print(f"❌ No camera at index {i}")
