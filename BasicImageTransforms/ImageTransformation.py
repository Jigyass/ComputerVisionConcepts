import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import image

def flipflopim(image):
    flipped_image = image [:, ::-1] #Using python's built in slicing functionality, we select all the rows to be in the same order, however the columns are selected in reverse order using -1
    flipfloppedimage = flipped_image[::-1, :] #Using python's slicing ability to flip the rows in the array while keeping the columns same since the passed image is already flipped


