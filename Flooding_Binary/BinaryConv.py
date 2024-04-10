'''
@author: Jigyas Sharma
@organization: University of Kansas
@Course: EECS 841(Computer Vision)
'''
# Import Necessary Libraries
from PIL import Image
import numpy as np

'''
@function_description: Calculates the optimal threshold for binarization using Otsu's method. Otsu's method selects a threshold that minimizes the intra-class variance, or equivalently, maximizes the inter-class variance.    
@param: Numpy array of the grayscale image.
@return: Optimal threshold value as an integer.
'''
def calculate_otsu_threshold(image_array):
    # Compute histogram and probabilities of each intensity level
    histogram, bins = np.histogram(image_array.flatten(), 256, [0, 256])
    total_pixels = image_array.size
    current_max, threshold = 0, 0
    sum_total, sum_foreground = 0, 0
    weight_background, weight_foreground = 0, 0
    
    # Compute total sum (used for mean calculation)
    for i in range(256):
        sum_total += i * histogram[i]
    
    # Iterate through all possible thresholds to find the best
    for i in range(256):
        weight_background += histogram[i]
        if weight_background == 0:
            continue
        weight_foreground = total_pixels - weight_background
        if weight_foreground == 0:
            break
        
        sum_foreground += i * histogram[i]
        
        mean_background = sum_foreground / weight_background
        mean_foreground = (sum_total - sum_foreground) / weight_foreground
        
        # Calculate between-class variance
        variance_between = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        # Check if new maximum found and update threshold
        if variance_between > current_max:
            current_max = variance_between
            threshold = i
            
    return threshold

'''
@function_description: Converts a grayscale image to binary using the optimal threshold calculated by Otsu's method.
@param: File path of the grayscale image.
@return: Converted Binary Image(Saved to current directory)
'''
def convert_to_binary(image_path):
    # Load the image and ensure it is in grayscale
    image = Image.open(image_path)
    image = image.convert('L')

    # Convert image to a numpy array for processing
    image_array = np.asarray(image)

    # Calculate optimized threshold using Otsu's method
    threshold = calculate_otsu_threshold(image_array)
    print(f"Optimized Threshold: {threshold}")

    # Apply the calculated threshold to binarize the image
    binary_image_array = np.where(image_array > threshold, 255, 0).astype(np.uint8)

    # Convert the binary array back to a PIL image and save it
    binary_image = Image.fromarray(binary_image_array)
    binary_image.save('binary_tools_optimized.png')

    print("Binary image has been saved as 'binary_tools_optimized.png'.")

#Call function to convert binary
#UBUNTU_PATH
convert_to_binary('/home/darksst/Desktop/ComputerVisionConcepts/Flooding_Binary/tools.pgm')
