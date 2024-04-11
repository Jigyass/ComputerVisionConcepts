import cv2
import numpy as np

def flood_fill(seed_point, fill_color, mask, c_img):
    """
    Flood fills the area connected to seed_point in mask with fill_color in c_img.
    """
    # Queue for BFS
    queue = [seed_point]
    # Set for keeping track of visited pixels
    visited = set()

    # Image dimensions
    h, w = mask.shape

    while queue:
        x, y = queue.pop(0)

        if (x, y) not in visited:
            visited.add((x, y))
            c_img[x, y] = fill_color  # Apply new color
            mask[x, y] = 0  # Update mask to prevent revisiting

            # Neighbors: up, down, left, right
            neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]

            for nx, ny in neighbors:
                if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 255:
                    queue.append((nx, ny))

# Read Image
img = cv2.imread('./tools.pgm', 0)

# Thresholding to create a binary inverse mask
_, mask = cv2.threshold(img, 155, 255, cv2.THRESH_BINARY_INV)

# Convert mask to a 3-channel image to apply colors
c_img = cv2.merge([mask, mask, mask])

# Base color for flood fill
fill_color = [100, 255, 200]

# Iterate over each pixel in the mask
for x in range(mask.shape[0]):
    for y in range(mask.shape[1]):
        # If the pixel is part of an object
        if mask[x, y] == 255:
            flood_fill((x, y), fill_color, mask, c_img)
            # Change color for next object; ensure values stay within valid range
            fill_color = [(v - 10) % 256 for v in fill_color]

# Rest of the flood fill code as before

# Save the output images
cv2.imwrite('original_image.png', img)
cv2.imwrite('binary_inverse_mask.png', mask)  # Optional: save the binary inverse mask
cv2.imwrite('colored_image.png', c_img)

