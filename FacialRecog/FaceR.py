'''
@author: Jigyas Sharma
@organization: University of Kansas
@Course: EECS 841(Computer Vision)
'''
#Library Imports
import cv2
import numpy as np
import os

def getImages(dir_path):
    """
    Load grayscale images from the specified directory path.

    Args:
    - dir_path: Path to the directory containing the images.

    Returns:
    - images: List of grayscale images.
    """
    images = []
    for file in os.listdir(dir_path):
        img = cv2.imread(os.path.join(dir_path, file), cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Error: Unable to load image {file}")
        else:
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
gallery_set = getImages('./GallerySet')
probe_set = getImages('./ProbeSet')

#Calculate Score Matrix by calling the function
Score_Matrix = []
Score_Matrix = calculateScoreMatrix(gallery_set, probe_set)
Score_Matrix = np.array(Score_Matrix)
Score_Array = Score_Matrix.reshape(100,100)
#Print the sub 10x10 matrix for validation
print(f'Score Matrix A:\n{Score_Array[0:9, 0:9]}')
#Calculate the d prime value
d_prime = calculate_d_prime(Score_Array)
print("d' value:", d_prime)
