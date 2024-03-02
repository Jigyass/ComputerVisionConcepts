import numpy as np
import matplotlib.pyplot as plt
import cv2  # Import OpenCV

# Function to read an image and convert to grayscale using OpenCV
def read_image_grayscale(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Read directly in grayscale
    return image

# Function to calculate histogram
def calculate_histogram(image):
    image_flat = image.flatten()
    values, counts = np.unique(image_flat, return_counts=True)
    return values, counts

# Paths to the images
swan_image_path = '/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/swan.pgm'
tools_image_path = '/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/tools.pgm'

# Read and process images
swan_image = read_image_grayscale(swan_image_path)
tools_image = read_image_grayscale(tools_image_path)
swan_values, swan_counts = calculate_histogram(swan_image)
tools_values, tools_counts = calculate_histogram(tools_image)

# Plotting the histograms
plt.figure(figsize=(14, 6))

# Swan Image Histogram
plt.subplot(1, 2, 1)
plt.bar(swan_values, swan_counts, color='gray')
plt.title('Histogram of Swan Image')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim(0, 255)

# Tools Image Histogram
plt.subplot(1, 2, 2)
plt.bar(tools_values, tools_counts, color='gray')
plt.title('Histogram of Tools Image')
plt.xlabel('Intensity Value')
plt.ylabel('Frequency')
plt.xlim(0, 255)

plt.tight_layout()
plt.show()

