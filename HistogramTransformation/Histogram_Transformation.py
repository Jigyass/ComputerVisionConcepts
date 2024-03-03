'''
@author: Jigyas Sharma
@organization: University of Kansas
@Course: EECS 841(Computer Vision)
'''

#Import all the necessary libraries here

import cv2
import matplotlib.pyplot as plt

'''
@param: path to an image
@output: Grayscale Image
@description: This function extracts an image from the defined system file path and opens it using cv2 in grayscale
'''
def Readimage(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    return image
'''
@param: CV2 Grayscale image
@output: Array where index is pixel intensity and value at index is the frequency of that pixel intensity
@description: This function calculates the frequency for each pixel intensity and stores it in an array for histogram plotting.
'''
def Histogram(image):
    histogram = [0] * 256 # Initializing an array with 256 zeroes as the pixel intensity should range from 0-255
    for row in image: #iterate over each row
        for pixel in row: #each pixel in the row
            histogram[pixel] += 1 #Array position is the pixel intensity value and it increments one for each instance of that intensity
    return histogram
'''
@param: Array(Histogram Frequencies), String(Title of the plotted histogram)
@output: plots a histogram
@description: This function converts the passed array to a histogram plot.
'''
def PlotHistogram(histogram, title):
    plt.bar(range(256), histogram, color='gray')
    plt.title(title)
    plt.xlabel('Intensity_Value')
    plt.ylabel('Frequency')
    plt.xlim(0, 255)
'''
@param: CV2 Image
@output: Equalized Image
@description: This function equalises the image using histogram equalization
'''
def Equalize(histogram):
    total_pixels = sum(histogram)
    pdf = [count / total_pixels for count in histogram]
    cdf = [0] * len(pdf)
    for i in range(len(pdf)):
        cdf[i] = pdf[i] + (cdf[i-1] if i>0 else 0)
    i=255
    while i>=0:
        if histogram[i] != 0:
            kmax = i
            break
        i -= 1
    equalized_histogram = [cdf[i] * kmax for i in range(len(cdf))]
    return equalized_histogram
'''
@param: String(System path to image), String(Name of the image)
@output: Histogram Plot
@description: This function runs all the functions to plot the frequency histogram for a given image.
'''
def RunHistogram(image_path, title):
    Image = Readimage(image_path)
    histogram = Histogram(Image)
    PlotHistogram(histogram, title)
    plt.show()
    equalized = Equalize(histogram)
    PlotHistogram(equalized, "Equalized")
    plt.show()



RunHistogram('/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/swan.pgm', 'SWAN')
RunHistogram('/home/darksst/Desktop/ComputerVisionConcepts/HistogramTransformation/tools.pgm', 'TOOLS')
