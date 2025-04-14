import cv2
import numpy as np
import matplotlib.pyplot as plt

# === REPLACE THESE with your actual calibration results ===
# Camera matrix (intrinsic parameters)
camera_matrix = np.array([[2.46519797e+03, 0.00000000e+00, 2.14569514e+02],
                          [0.00000000e+00, 3.32173460e+03, 2.84260340e+02],
                          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# Distortion coefficients
dist_coeffs = np.array([[-1.36072952e+01,  6.76650039e+02,  1.84247240e-02, -2.21974511e-02, -1.05890887e+04]])

"""
camera_matrix = np.array([[2.46519797e+03, 0.00000000e+00, 2.14569514e+02],
 [0.00000000e+00, 3.32173460e+03, 2.84260340e+02],
 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
dist_coeffs = np.array([[-1.36072952e+01,  6.76650039e+02,  1.84247240e-02, -2.21974511e-02,
  -1.05890887e+04]])

"""

# === Load a test image ===
img = cv2.imread('calibration_images/image_01.png')  # replace with the actual image path
if img is None:
    print("Error: Image could not be loaded. Please check the file path.")
else:
    # Image dimensions
    h, w = img.shape[:2]

    # === Undistort the image ===
    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)

    # === Compute absolute difference for visualization ===
    # Convert both images to grayscale for comparison
    gray_orig = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_undist = cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2GRAY)
    diff = cv2.absdiff(gray_orig, gray_undist)

    # === Display the images and difference map ===
    plt.figure(figsize=(15, 5))

    # Original Distorted Image
    plt.subplot(1, 3, 1)
    plt.title("Original (Distorted)")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Undistorted Image
    plt.subplot(1, 3, 2)
    plt.title("Undistorted")
    plt.imshow(cv2.cvtColor(undistorted_img, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    # Difference Map (Grayscale Heatmap)
    plt.subplot(1, 3, 3)
    plt.title("Difference (Grayscale Heatmap)")
    plt.imshow(diff, cmap='hot')
    plt.axis("off")

    # Show the images
    plt.tight_layout()
    plt.show()

    # === Calculate Image Comparison Metrics ===

    # Mean Squared Error (MSE)
    def mse(image1, image2):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        error = np.sum((image1 - image2) ** 2)
        mse_value = error / float(image1.shape[0] * image1.shape[1])
        return mse_value

    # Peak Signal-to-Noise Ratio (PSNR)
    def psnr(image1, image2):
        mse_value = mse(image1, image2)
        if mse_value == 0:
            return 100  # Perfect match (no error)
        PIXEL_MAX = 255.0
        psnr_value = 20 * np.log10(PIXEL_MAX / np.sqrt(mse_value))
        return psnr_value

    # Structural Similarity Index (SSIM)
    from skimage.metrics import structural_similarity as ssim
    def ssim_value(image1, image2):
        # Convert to grayscale
        gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        
        # Compute SSIM with explicit parameters
        ssim_index, _ = ssim(gray1, gray2, win_size=3, channel_axis=None, full=True)
        return ssim_index

    # Histogram Comparison
    def histogram_comparison(image1, image2):
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
        hist1 = cv2.calcHist([image1], [0], None, [256], [0, 256])
        hist2 = cv2.calcHist([image2], [0], None, [256], [0, 256])
        hist_corr = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        return hist_corr

    # Calculate the comparison metrics
    mse_value = mse(img, undistorted_img)
    psnr_value = psnr(img, undistorted_img)
    ssim_index = ssim_value(img, undistorted_img)
    hist_corr_value = histogram_comparison(img, undistorted_img)

    # Print out the results
    print(f"MSE: {mse_value}")
    print(f"PSNR: {psnr_value}")
    print(f"SSIM: {ssim_index}")
    print(f"Histogram Correlation: {hist_corr_value}")

    # Optional: Save the difference image
    cv2.imwrite('difference_image.png', diff)
    print("Difference image saved as 'difference_image.png'")
