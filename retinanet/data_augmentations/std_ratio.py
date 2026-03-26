import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
from scipy.ndimage import gaussian_filter1d

## This program takes images and performs colour correction on them
## The method involves finding the standard deviation ratio of the R, G, and B channels
## and then adjusting the values to produces an image closer to the true colours
def compute_mean_std(image):
    #Compute mean and standard deviation per channel
    mean = np.mean(image, axis=(0, 1))
    std = np.std(image, axis=(0, 1))
    return mean, np.maximum(std,0.000001)  # Avoid division by zero

def improved_white_balance(image):
    #Apply improved underwater white balance method
    image = image.astype(np.float32) / 255.0  # Normalize
    
    # Split channels
    r, g, b = cv2.split(image)
    
    # Compute mean and std per channel
    mean_c, std_c = compute_mean_std(image)
    mean_g, std_g = np.mean(g), np.std(g)  # Reference green channel
    
    # Apply the first equation to each channel
    corrected_channels = []
    for channel, mean, std in zip([r, g, b], mean_c, std_c):
        adjusted = channel + (std_g / std) * (mean_g - mean) * (1 - channel) * g
        corrected_channels.append(adjusted)
    
    # Merge back
    corrected_image = cv2.merge(corrected_channels)
    
    # Apply compensation equation
    mean_corrected = np.mean(corrected_image, axis=(0, 1))
    final_channels = [ch - mean_corrected[i] + mean_g for i, ch in enumerate(corrected_channels)]
    
    # Merge and clip values to [0,1]
    final_image = cv2.merge(final_channels)
    final_image = np.clip(final_image, 0, 1)

    # Convert back to 8-bit
    final_image = (final_image * 255).astype(np.uint8)

    return final_image

# image path
def SDR(image_path):
    #load in one image at a time
    if image_path.endswith(".jpg"):
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        corrected_img = improved_white_balance(img)

        # Split channels
        r, g, b = cv2.split(img)
        corrected_r, corrected_g, corrected_b = cv2.split(corrected_img)
        
        ##comment this out when running from preprocessing.py
        # display the original and corrected images
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img)
        ax[0].set_title("Original Image")
        ax[0].axis("off")
        ax[1].imshow(corrected_img)
        ax[1].set_title("Corrected Image")
        ax[1].axis("off")
        plt.show()

        # Plot and display the histograms of the original and corrected images
        fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Create 1 row, 2 columns of subplots

        # Compute histograms for the original image
        hist_r = np.histogram(r.ravel(), bins=256, range=(0, 256))[0]
        hist_g = np.histogram(g.ravel(), bins=256, range=(0, 256))[0]
        hist_b = np.histogram(b.ravel(), bins=256, range=(0, 256))[0]

        # Apply Gaussian smoothing to the histograms
        hist_r_smooth = gaussian_filter1d(hist_r, sigma=2)
        hist_g_smooth = gaussian_filter1d(hist_g, sigma=2)
        hist_b_smooth = gaussian_filter1d(hist_b, sigma=2)

        # Plot smoothed histograms for the original image
        ax[0].plot(np.arange(256) / 255, hist_r_smooth, color="red", label="Red Channel (Smoothed)")
        ax[0].plot(np.arange(256) / 255, hist_g_smooth, color="green", label="Green Channel (Smoothed)")
        ax[0].plot(np.arange(256) / 255, hist_b_smooth, color="blue", label="Blue Channel (Smoothed)")
        ax[0].set_title("Original Image")
        ax[0].set_xlabel("Normalized Intensity")
        ax[0].set_ylabel("Frequency")
        ax[0].set_xscale("log")
        ax[0].legend()

        # Compute histograms for the corrected image
        hist_corrected_r = np.histogram(corrected_r.ravel(), bins=256, range=(0, 256))[0]
        hist_corrected_g = np.histogram(corrected_g.ravel(), bins=256, range=(0, 256))[0]
        hist_corrected_b = np.histogram(corrected_b.ravel(), bins=256, range=(0, 256))[0]

        # Apply Gaussian smoothing to the histograms
        hist_corrected_r_smooth = gaussian_filter1d(hist_corrected_r, sigma=2)
        hist_corrected_g_smooth = gaussian_filter1d(hist_corrected_g, sigma=2)
        hist_corrected_b_smooth = gaussian_filter1d(hist_corrected_b, sigma=2)

        # Plot smoothed histograms for the corrected image
        ax[1].plot(np.arange(256) / 255, hist_corrected_r_smooth, color="red", label="Red Channel (Smoothed)")
        ax[1].plot(np.arange(256) / 255, hist_corrected_g_smooth, color="green", label="Green Channel (Smoothed)")
        ax[1].plot(np.arange(256) / 255, hist_corrected_b_smooth, color="blue", label="Blue Channel (Smoothed)")
        ax[1].set_title("Corrected Image")
        ax[1].set_xlabel("Normalized Intensity")
        ax[1].set_ylabel("Frequency")
        ax[1].set_xscale("log")
        ax[1].legend()

        plt.tight_layout()
        plt.show()

        # Calculate mean and std for original and corrected images
        mean_orig, std_orig = compute_mean_std(img)
        mean_corr, std_corr = compute_mean_std(corrected_img)

        # Print table of before and after mean and std for each channel
        print("\nChannel Statistics (Original vs Corrected):")
        print("{:<10} {:>10} {:>10} {:>15} {:>15}".format("Channel", "Mean (Orig)", "Mean (Corr)", "Std (Orig)", "Std (Corr)"))
        for i, ch in enumerate(["Red", "Green", "Blue"]):
            print("{:<10} {:>10.2f} {:>10.2f} {:>15.2f} {:>15.2f}".format(
                ch, mean_orig[i], mean_corr[i], std_orig[i], std_corr[i]
            ))

        return corrected_img
    
if __name__  == "__main__":
    image_path = "D:/Patrick PAST UNI STUFF/Honours/Handfish Detections - Human - Tight Boxes/PR_20220907_013611_959_FC16.jpg"
    #image_path = "C:/Users/pmqui/OneDrive/Desktop/Honours/Thesis/Birdie.jpg"
    SDR(image_path)