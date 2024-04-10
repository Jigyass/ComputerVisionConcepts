from PIL import Image
import numpy as np
import random

def flood_fill(image, x, y, newColor, visited):
    """
    Recursively fills connected components of the image starting from (x, y).
    """
    # Base case checks
    if x < 0 or y < 0 or x >= image.shape[0] or y >= image.shape[1]:
        return
    if visited[x, y]:  # Only proceed if the pixel hasn't been visited
        return

    # Mark the pixel as visited
    visited[x, y] = True
    
    # Color the pixel in the RGB image
    image[x, y] = newColor
    
    # Recursive calls for the four neighbors
    flood_fill(image, x+1, y, newColor, visited)
    flood_fill(image, x-1, y, newColor, visited)
    flood_fill(image, x, y+1, newColor, visited)
    flood_fill(image, x, y-1, newColor, visited)

def color_objects(binary_image_path):
    """
    Colors each separate object in a binary image with a unique color.
    """
    binary_image = Image.open(binary_image_path)
    binary_image = binary_image.convert('L')  # Convert to grayscale to ensure it's single-channel
    binary_array = np.array(binary_image)

    # Initialize a visited array to keep track of which pixels have been processed
    visited = np.zeros_like(binary_array, dtype=bool)
    
    # Prepare an output RGB image array
    output_image = np.zeros((*binary_array.shape, 3), dtype=np.uint8)
    
    for x in range(binary_array.shape[0]):
        for y in range(binary_array.shape[1]):
            if binary_array[x, y] == 255 and not visited[x, y]:  # Check if the pixel is part of an object and not visited
                newColor = [random.randint(0, 255) for _ in range(3)]  # Generate a random color for each object
                flood_fill(output_image, x, y, newColor, visited)  # Perform flood fill from this pixel

    # Convert the colored array back to an image and save it
    colored_image = Image.fromarray(output_image)
    colored_image.save('colored_objects.png')
    print("Colored image has been saved as 'colored_objects.png'.")

# Example usage: Adjust the path to your binary image file.
color_objects('/home/darksst/Desktop/ComputerVisionConcepts/Flooding_Binary/binary_tools_optimized.png')

