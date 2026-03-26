## @author - Patrick Quinlan
## @date - December 19 2024
## @descripting - This program contains functions to augment images to artificially enlarge a dataset in multiple different ways
##                or enhance the supplied data

import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import statistics
import cv2
from imgaug import augmenters as iaa
#import matplotlib.image as mpimg

def get_average_color(image):
    """
    Calculate the average color value of an image.
    
    :param image: PIL Image object
    :return: Tuple representing the average color (R, G, B)
    """
    # Convert the image to a NumPy array
    image_array = np.asarray(image)
    
    # Ensure the image is in RGB format
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        # Calculate the mean for each channel (R, G, B)
        avg_color = tuple(image_array.mean(axis=(0, 1)).astype(int))
        return avg_color
    else:
        raise ValueError("Image must be in RGB format.")
    
#Random Erasing Augmentations
#Due to nature of specimens being small in screen may be more beneficial to manually erase rectangles near handfish
def random_erasing(image):
    image_copy = image.copy()
    #get average colour of image
    pixelAve = get_average_color(image_copy)

    #choose random rectangle size
    max_size_height = int(image.shape[0]/10) #image height
    max_size_width = int(image.shape[1]/10) #image width
    max_size_width, max_size_height = image_copy.size #image height
    num_rectangles = 10#random.randint(1,5)
    try:
        for i in range(0,num_rectangles):
            width = random.randint(1, int(max_size_width)) #rectangle width
            height = random.randint(1, int(max_size_height)) #rectangle height
            corner_start_x = random.randint(0,width) #corner start on x coordinate
            corner_start_y = random.randint(0,height)
            for y in range(corner_start_y,corner_start_y+height):  
                y = int(y)  
                if y>=height:
                    break
                for x in range(corner_start_x,corner_start_x+width):
                    x=int(x)
                    if x>=width:
                        break
                    image_copy.putpixel( (x,y), (pixelAve,pixelAve,pixelAve))
        return image_copy
    except Exception as error:
        print("ERROR", error)

#Making colour corrections for red light deficiency in images
# T
def STD_colour_correction():
    #TODO
    return

#Making a mosaic of multiple images
def mosaic():
    #TODO
    return

#Resizing and reshaping images
def resize():
    #TODO
    return

#adding noise to the images
def add_noise(image):
    image_copy = image.copy()
    image_copy = np.asarray(image)
    if image_copy.max() <= 1:
        image_copy = (image_copy * 255).astype(np.uint8)
    row,col,ch= image_copy.shape
    mean = 0
    sigma = 20
    gauss = np.random.normal(mean,sigma,(row,col,ch))
    gauss = gauss.reshape(row,col,ch)
    noisy = image_copy + gauss
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy

def noisy_image_test():
    test_jpg = "media_collection/Handfish Detections - Human and AI generated/PR_20090615_232832_252_LC16.jpg"
    test_image = Image.open(test_jpg)
    augmented_image_data = add_noise(test_image)
    augmented_image = Image.fromarray(augmented_image_data, 'RGB')
    augmented_image.show()


def random_erasing_test():
    test_jpg = "media_collection/Handfish Detections - Human and AI generated/PR_20090615_232832_252_LC16.jpg"
    test_image = Image.open(test_jpg)
    print(test_image.size)
    print("===============")
    (1360, 1024)
    I = np.asarray(test_image)
    I.shape
    (1024,1360,3)
    print(I)
    augmented_image = random_erasing(test_image)
    print("\n========================\n")
    augmented_image.show()

def main():
    #random_erasing_test()
    noisy_image_test()

if __name__ == "__main__":
    main()