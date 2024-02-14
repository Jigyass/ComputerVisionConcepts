import tkinter as tk
from tkinter import filedialog
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

'''
@param: Numpy Image Array
@output: Numpy Image Array
@description: This function uses slicing to mirror an image and flip it vertically to attain a flip flopped image. 
'''
def flipflopim(image):
    flipped_image = image [:, ::-1] #Using python's built in slicing functionality, we select all the rows to be in the same order, however the columns are selected in reverse order using -1
    flipfloppedimage = flipped_image[::-1, :] #Using python's slicing ability to flip the rows in the array while keeping the columns same since the passed image is already flipped

'''
@param: None
@output: 2 pillow images
@description: The function asks the user to select an image using tkinter dialog box. The image is processed as a pillow image which is converted to a numpy array for transformations. The function then displays the 2 pillow images, the original and the transformed image.
'''
def ImageProcess_Select():
    #Set up tkinter permissions for dialog and select an image
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select an image",
        filetypes=(("Image files", "*.jpg;*.jpeg;*.png;*.gif"), ("All files", "*.*"))
    )
    #Check if file exists
    if not file_path:
        print("File does not exist")
        return

    #Load the selected image and process using pillow and convert to a numpy array
    pillow_image = Image.open(file_path)
    numpy_image = np.array(pillow_image)

    #Transform the image using the flipflopim function
    TransformedI = flipflopim(numpy_image)

    #Convert the image back to a pillow image for display
    TransformedI_pillow = Image.fromarray(TransformedI)

    #Display the original image and the transformed Image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(pillow_image)
    plt.title("Original Image")
    plt.axis('off')  # Hide axes for better visualization

    plt.subplot(1, 2, 2)
    plt.imshow(TransformedI_pillow)
    plt.title("Processed Image")
    plt.axis('off')  # Hide axes for better visualization

    plt.show()

ImageProcess_Select() #Function Call to start the process
