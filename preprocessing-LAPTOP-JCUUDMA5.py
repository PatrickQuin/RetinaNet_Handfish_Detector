# retrain using novel dataset using all data
# to find a threshold of performance. This program will use resized
# images so that there is a more consistent distance from the camera
import random
from ultralytics import YOLO
import json
from PIL import Image, ImageDraw
import os
from sklearn.model_selection import KFold
import yaml
import logging
from imgaug.augmentables.kps import KeypointsOnImage, Keypoint
import shutil
import numpy as np
import cv2

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


class KeypointsWithFilename:
    def __init__(self, keypoints, shape, filename):
        self.keypoints_on_image = KeypointsOnImage(keypoints, shape)
        self.filename = filename

def convert_to_yolo_format(keypoints_on_image, img_width, img_height):
    keypoints = keypoints_on_image.keypoints
    if len(keypoints) == 0:
        return ""
    keypoint = keypoints[0]
    x_center = keypoint.x
    y_center = keypoint.y
    width = 0.1  
    height = 0.1 
    class_id = 0 
    return f"{class_id} {x_center} {y_center} {width} {height}"

def process_data():
    # Load the unaltered dataset
    unaltered_data_path = "C:/Users/pmqui/OneDrive - University of Tasmania/Honours/Repositories/Handfish Detections - Human - Tight Boxes"
    yolo_ann_path = "yolo_data/training"
    dataset = []
    annotations = []
    for image_name in os.listdir(unaltered_data_path):
        if image_name.endswith(".jpg"):
            img_path = os.path.join(unaltered_data_path, image_name)
            dataset.append(img_path)
    # for image_name in os.listdir(unaltered_data_path):
    #     if image_name.endswith(".xml"):
    #         annot_path = os.path.join(unaltered_data_path, image_name)
    #         annotations.append(annot_path)
        
    #create a train test split
    train_val_dataset = []
    test_dataset = []
    #train_val_annotations = []
    #test_annotations = []
    for index, img in enumerate(dataset):
        if (random.randint(0,1000)/1000 < 0.85):
            train_val_dataset.append(img)
            #train_val_annotations.append(annotations[index])
        else:
            test_dataset.append(img)
            #test_annotations.append(annotations[index])
    
     # create a k fold split
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    for fold, (train_index, val_index) in enumerate(kf.split(train_val_dataset)):
        fold_train_dataset = [train_val_dataset[i] for i in train_index]
        fold_val_dataset = [train_val_dataset[i] for i in val_index]
        # fold_train_annotations = [train_val_annotations[i] for i in train_index]
        # fold_val_annotations = [train_val_annotations[i] for i in val_index]

        im_fold_dir = f"YOLO/data/fold_{fold}/images"
        ann_fold_dir = f"YOLO/data/fold_{fold}/labels"
        os.makedirs(im_fold_dir, exist_ok=True)
        os.makedirs(ann_fold_dir, exist_ok=True)

        im_fold_train_path = os.path.join(im_fold_dir, "train")
        im_fold_val_path = os.path.join(im_fold_dir, "val")
        ann_fold_train_path = os.path.join(ann_fold_dir, "train")
        ann_fold_val_path = os.path.join(ann_fold_dir, "val")
        os.makedirs(im_fold_train_path, exist_ok=True)
        os.makedirs(im_fold_val_path, exist_ok=True)
        os.makedirs(ann_fold_train_path, exist_ok=True)
        os.makedirs(ann_fold_val_path, exist_ok=True)

        #save train images
        for img in fold_train_dataset:
            shutil.copy(img, im_fold_train_path)
        #save train annotations
        for image_name in os.listdir(im_fold_train_path):
            for annot_name in os.listdir(yolo_ann_path):
                if annot_name[:-3] == image_name[:-3]:
                    if annot_name.endswith(".txt"):
                        shutil.copy(os.path.join(yolo_ann_path,annot_name), ann_fold_train_path)

        #save train images
        for img in fold_val_dataset:
            shutil.copy(img, im_fold_val_path)

        for image_name in os.listdir(im_fold_val_path):
            for annot_name in os.listdir(yolo_ann_path):
                if annot_name[:-3] == image_name[:-3]:
                    if annot_name.endswith(".txt"):
                        shutil.copy(os.path.join(yolo_ann_path,annot_name), ann_fold_val_path)

    im_test_path = f"YOLO/data/test/images"
    ann_test_path = f"YOLO/data/test/labels"
    os.makedirs(im_test_path, exist_ok=True)
    os.makedirs(ann_test_path, exist_ok=True)
    #save test images
    for img in test_dataset:
        shutil.copy(img, im_test_path)
    #save test annotations
        for image_name in os.listdir(im_test_path):
            for annot_name in os.listdir(yolo_ann_path):
                if annot_name[:-3] == image_name[:-3]:
                    if annot_name.endswith(".txt"):
                        shutil.copy(os.path.join(yolo_ann_path,annot_name), ann_test_path)



if __name__ == "__main__":
    process_data()
