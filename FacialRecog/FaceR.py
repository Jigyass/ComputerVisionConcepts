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
        img = cv2.imread(f'{dir_path}/{file}', 0)
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
    - score_matrix: Matrix containing similarity scores between probe and gallery images.
    """
    score_matrix = np.zeros((len(probe_set), len(gallery_set)))
    for i in range(len(probe_set)):
        for j in range(len(gallery_set)):
            score_matrix[i, j] = normalizedCorrCoeff(probe_set[i], gallery_set[j])
    return score_matrix

def calculate_d_prime(score_matrix):
    """
    Calculate the d' value for face recognition performance evaluation.

    Args:
    - score_matrix: Matrix containing similarity scores between probe and gallery images.

    Returns:
    - d_prime: Decidability index value.
    """
    gal_score = []
    prob_score = []
    for i in range(len(score_matrix)):
        for j in range(len(score_matrix)):
            if i == j:
                gal_score.append(score_matrix[i, j])
            else:
                prob_score.append(score_matrix[i, j])

    mu1 = np.mean(gal_score)
    mu0 = np.mean(prob_score)
    sigma1 = np.std(gal_score)
    sigma0 = np.std(prob_score)

    d_prime = np.sqrt(2) * np.abs(mu1 - mu0) / np.sqrt(sigma1**2 + sigma0**2)
    return d_prime

# Load images from directories
gallery_set = getImages('./GallerySet')
probe_set = getImages('./ProbeSet')

# Calculate score matrix
score_matrix = calculateScoreMatrix(gallery_set, probe_set)
print(f'Score Matrix A:\n{score_matrix[0:9, 0:9]}')

# Calculate d' value
d_prime = calculate_d_prime(score_matrix)
print(f'd\' value: {d_prime}')

