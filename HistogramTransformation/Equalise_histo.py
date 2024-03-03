import cv2  # Import OpenCV
import numpy as np
import matplotlib.pyplot as plt 


# Existing functions for reading an image and calculating histogram
def read_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image

# Manual Histogram Equalization Function
def manual_histogram_equalization(image):
    histogram = [0] * 256
    for row in image:
        for pixel in row:
            histogram[pixel] += 1

    total_pixels = len(image) * len(image[0])
    pdf = [x / total_pixels for x in histogram]

    cdf = []
    running_sum = 0
    for p in pdf:
        running_sum += p
        cdf.append(running_sum)

    new_intensity_map = {i: int(cdf[i] * 255) for i in range(len(cdf))}

    equalized_image = np.array([[new_intensity_map[pixel] for pixel in row] for row in image], dtype=np.uint8)
    return equalized_image

def display_equalized_image(image_path):
    # Read the image
    original_image = read_image_grayscale(image_path)
    
    # Apply manual histogram equalization
    equalized_image = manual_histogram_equalization(original_image)
    
    # Display the original and equalized image using Matplotlib
    plt.figure(figsize=(10, 4))
    
    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')  # Hide axis
    
    # Display equalized image
    plt.subplot(1, 2, 2)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('Equalized Image')
    plt.axis('off')  # Hide axis
    
    plt.show()

display_equalized_image('/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/swan.pgm')
display_equalized_image('/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/tools.pgm')
