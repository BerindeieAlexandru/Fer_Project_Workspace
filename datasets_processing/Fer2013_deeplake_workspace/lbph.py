import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern
from scipy.spatial import ConvexHull, QhullError

def compute_lbp(image, radius=3, n_points=8):
    """
    Compute Local Binary Pattern (LBP) for the given image.

    Parameters:
        image (numpy.ndarray): Grayscale image.
        radius (int): Radius of the LBP pattern.
        n_points (int): Number of circularly symmetric neighbor points.

    Returns:
        numpy.ndarray: LBP image.
    """
    # Compute LBP
    lbp = local_binary_pattern(image, n_points, radius, method="uniform")
    return lbp

def display_images_side_by_side(original, lbp_image):
    """
    Display the original and LBP images side by side.

    Parameters:
        original (numpy.ndarray): Original grayscale image.
        lbp_image (numpy.ndarray): LBP-transformed image.
    """
    plt.figure(figsize=(10, 5))

    # Original Image
    plt.subplot(1, 2, 1)
    plt.imshow(original, cmap='gray')
    plt.title("Original Image")
    plt.axis("off")

    # LBP Image
    plt.subplot(1, 2, 2)
    plt.imshow(lbp_image, cmap='gray')
    plt.title("LBP Image")
    plt.axis("off")

    plt.tight_layout()
    plt.show()

# Load an example image (grayscale)
image_path = r"D:\Alex\Desktop\datasets_processing\affectnet_workspace\affectnet\anger\ffhq_174.png"
original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

# Check if the image is loaded properly
if original_image is None:
    raise FileNotFoundError(f"Image not found at {image_path}")

resized_image = cv2.resize(original_image, (256, 256))
# Compute LBP image
lbp_image = compute_lbp(resized_image)
lbp_image_normalized = cv2.normalize(lbp_image, None, 0, 255, cv2.NORM_MINMAX)
lbp_image_normalized = np.uint8(lbp_image_normalized)

# Display images side by side
display_images_side_by_side(resized_image, lbp_image_normalized)