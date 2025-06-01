import cv2
import os

target_dir = "data/gray_bg_channel/val"
os.makedirs(target_dir, exist_ok=True)
img_dir = "data/val"

for filename in os.listdir(img_dir):
    if filename.endswith(".xml"):
        continue  # Skip XML files
    # Read the image
    image = cv2.imread(os.path.join(img_dir, filename))
    r,g,b = cv2.split(image)
    two_channel = cv2.merge((b,b,g,g))  # Create a 2-channel image with only the green channel
    cv2.imwrite(os.path.join(target_dir, filename), two_channel)

target_dir = "data/gray_bg_channel/test"
os.makedirs(target_dir, exist_ok=True)
img_dir = "data/test"

for filename in os.listdir(img_dir):
    if filename.endswith(".xml"):
        continue  # Skip XML files
    # Read the image
    image = cv2.imread(os.path.join(img_dir, filename))
    r,g,b = cv2.split(image)
    two_channel = cv2.merge((b,b,g,g))  # Create a 2-channel image with only the green channel
    cv2.imwrite(os.path.join(target_dir, filename), two_channel)

target_dir = "data/gray_bg_channel/train"
os.makedirs(target_dir, exist_ok=True)
img_dir = "data/train"

for filename in os.listdir(img_dir):
    if filename.endswith(".xml"):
        continue  # Skip XML files
    # Read the image
    image = cv2.imread(os.path.join(img_dir, filename))
    r,g,b = cv2.split(image)
    two_channel = cv2.merge((b,b,g,g))  # Create a 2-channel image with only the green channel
    cv2.imwrite(os.path.join(target_dir, filename), two_channel)