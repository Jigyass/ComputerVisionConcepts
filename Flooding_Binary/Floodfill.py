import cv2
import numpy as np

# Function to perform flood fill algorithm
def flood_fill(image, seed_point, fill_color):
    """
    Performs a flood fill algorithm on a given image starting from a seed point.
    
    :param image: Numpy array of the image.
    :param seed_point: Tuple (x, y) as the starting point for flood fill.
    :param fill_color: List [B, G, R] specifying the color used for filling.
    """
    h, w = image.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)  # Additional border to avoid boundary issues
    cv2.floodFill(image, mask, seed_point, fill_color)

# Read the image in grayscale
img = cv2.imread('./tools.pgm', 0)

# Apply threshold to create a binary image
_, binary_img = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY_INV)

# Convert binary image to 3-channel image to apply colors
colored_img = cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR)

# Define initial fill color
fill_color = [100, 255, 200]  # BGR format

# Iterate over each pixel in the binary image
for x in range(binary_img.shape[0]):
    for y in range(binary_img.shape[1]):
        # Check if the pixel is white (indicating it's part of the object to fill)
        if binary_img[x, y] == 255:
            # Perform flood fill from this pixel
            flood_fill(colored_img, (y, x), fill_color)  # Note: cv2 uses (y, x) for coordinates

            # Modify the fill color for the next object
            fill_color = [(c - 10) if c - 10 >= 0 else c for c in fill_color]

# Save the original binary image and the result
cv2.imwrite('binary_image.png', binary_img)
cv2.imwrite('filled_image.png', colored_img)

