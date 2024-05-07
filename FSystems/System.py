'''
@author: Jigyas Sharma
@organization: University of Kansas
@Course: EECS 841(Computer Vision)
'''
#Library Imports
import cv2
import numpy as np
import os

def Box_Filter(image):
    """
    Apply a box filter to an image to achieve a simple blur effect.

    This function uses a 3x3 box filter kernel, where each element is 1/9, to average the pixel values in the image.
    This averaging process helps in reducing high-frequency noise and smoothing the image.

    Args:
    - image: The input image to be filtered (np.array).

    Returns:
    - blurred_image: Image after applying the box filter (np.array).
    """
    box_kernel = np.array([[1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9],
                           [1/9, 1/9, 1/9]])
    box_image = cv2.filter2D(src=image, ddepth=-1, kernel=box_kernel)
    return box_image

def Gaussian_Filter(image):
    """
    Apply a Gaussian filter to an image to reduce noise and detail.

    This function uses a 3x3 Gaussian kernel which emphasizes the center pixel, weighted more than its neighbors.
    This kernel is particularly effective at maintaining edges while reducing noise.

    Args:
    - image: The input image (np.array).

    Returns:
    - filtered_image: Image after applying the Gaussian filter (np.array).
    """
    gaussian_kernel = np.array([[1, 2, 1],
                                [2, 4, 2],
                                [1, 2, 1]]) * (1/16)
    filtered_image = cv2.filter2D(src=image, ddepth=-1, kernel=gaussian_kernel)
    return filtered_image

def getImages(dir_path, preprocess=None):
    """
    Load images from the specified directory path and optionally apply a preprocessing function.

    Args:
    - dir_path: Path to the directory containing the images.
    - preprocess: Optional preprocessing function to apply to each image. If None, no preprocessing is applied.

    Returns:
    - images: List of images, processed if a preprocessing function is provided.
    """
    images = []
    for file in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to load image {file}")
        else:
            if preprocess:
                img = preprocess(img)
            images.append(img)
    return images

def normalizedCorrCoeff(img1: np.ndarray, img2: np.ndarray):
    """
    Calculate the normalized correlation coefficient between two images.

    Args:
    - img1: First image.
    - img2: Second image.

    Returns:
    - r: Normalized correlation coefficient.
    """
    x = img1.reshape((-1, 1))
    y = img2.reshape((-1, 1))
    xn = x - np.mean(x)
    yn = y - np.mean(y)
    r = (np.sum(xn * yn)) / (np.sqrt(np.sum(xn**2)) * np.sqrt(np.sum(yn**2)))
    return r

def calculateScoreMatrix(gallery_set, probe_set):
    """
    Calculate the score matrix for face recognition.

    Args:
    - gallery_set: List of images in the gallery set.
    - probe_set: List of images in the probe set.

    Returns:
    - score_matrix: One-dimensional array containing similarity scores between probe and gallery images.
    """
    score_matrix = []
    for i in probe_set:
        for j in gallery_set:
            temp = normalizedCorrCoeff(i, j)
            score_matrix.append(temp)
    return score_matrix



def calculate_d_prime(Score_Array):
    """
    Calculate the d' value for face recognition performance evaluation.

    Args:
    - score_matrix: Matrix containing similarity scores between probe and gallery images.

    Returns:
    - d_prime: Decidability index value.
    """
    gal_score = []
    prob_score = []
    for i in range(99):
        for j in range(99):
            temp = Score_Array[i, j]
            if i == j:
                gal_score.append(temp)
            else:
                prob_score.append(temp)

    mu1 = np.mean(gal_score)
    mu0 = np.mean(prob_score)
    sigma1 = np.std(gal_score)
    sigma0 = np.std(prob_score)

    d_prime = np.sqrt(2) * np.abs(mu1 - mu0) / np.sqrt(sigma1**2 + sigma0**2)
    return d_prime

# Load images from directories
gallery_set = getImages('./GallerySet', Box_Filter)
probe_set = getImages('./ProbeSet/ProbeSet', Box_Filter)

#Calculate Score Matrix by calling the function
Score_Matrix = []
Score_Matrix = calculateScoreMatrix(gallery_set, probe_set)
Score_Matrix = np.array(Score_Matrix)
Score_Array = Score_Matrix.reshape(100,100)
#Print the sub 10x10 matrix for validation
print(f'Score Matrix C:\n{Score_Array[0:9, 0:9]}')
#Calculate the d prime value
d_prime = calculate_d_prime(Score_Array)
print("d' value:", d_prime)
