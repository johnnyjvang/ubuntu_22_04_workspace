import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

# === Camera calibration results ===
camera_matrix = np.array([[2.46519797e+03, 0.00000000e+00, 2.14569514e+02],
                          [0.00000000e+00, 3.32173460e+03, 2.84260340e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

dist_coeffs = np.array([[-1.36072952e+01, 6.76650039e+02, 1.84247240e-02,
                         -2.21974511e-02, -1.05890887e+04]])

# === Image directory ===
image_folder = 'calibration_images'

# === Metrics storage ===
image_names = []
mse_values = []
psnr_values = []
ssim_values = []
hist_corr_values = []

# === Metric functions ===
def mse(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    error = np.sum((gray1 - gray2) ** 2)
    return error / float(gray1.shape[0] * gray1.shape[1])

def psnr(image1, image2):
    mse_val = mse(image1, image2)
    if mse_val == 0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * np.log10(PIXEL_MAX / np.sqrt(mse_val))

def ssim_value(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    score, _ = ssim(gray1, gray2, win_size=3, channel_axis=None, full=True)
    return score

def histogram_comparison(image1, image2):
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
    hist1 = cv2.calcHist([gray1], [0], None, [256], [0, 256])
    hist2 = cv2.calcHist([gray2], [0], None, [256], [0, 256])
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

# === Process all images ===
for filename in sorted(os.listdir(image_folder)):
    if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        img_path = os.path.join(image_folder, filename)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error loading image {filename}")
            continue

        # Undistort the image
        undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

        # Calculate metrics
        mse_val = mse(img, undistorted_img)
        psnr_val = psnr(img, undistorted_img)
        ssim_val = ssim_value(img, undistorted_img)
        hist_val = histogram_comparison(img, undistorted_img)

        # Store results
        image_names.append(filename)
        mse_values.append(mse_val)
        psnr_values.append(psnr_val)
        ssim_values.append(ssim_val)
        hist_corr_values.append(hist_val)

# === Plot the results ===
plt.figure(figsize=(16, 10))

plt.subplot(2, 2, 1)
plt.plot(image_names, mse_values, marker='o', color='blue')
plt.title('Mean Squared Error (MSE)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(image_names, psnr_values, marker='o', color='green')
plt.title('Peak Signal-to-Noise Ratio (PSNR)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(image_names, ssim_values, marker='o', color='red')
plt.title('Structural Similarity Index (SSIM)')
plt.xticks(rotation=45)
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(image_names, hist_corr_values, marker='o', color='purple')
plt.title('Histogram Correlation')
plt.xticks(rotation=45)
plt.grid(True)

plt.tight_layout()
plt.show()
