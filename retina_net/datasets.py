import shutil
import sys

import torch
import cv2
import numpy as np
import os
import glob as glob
from pathlib import Path
from xml.etree import ElementTree as et

# Allow running this file directly: `python retina_net/train.py`
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from retina_net.config import (
    CLASSES, RESIZE_TO, TRAIN_DIR, BATCH_SIZE
)
from torch.utils.data import Dataset, DataLoader
from retina_net.custom_utils import collate_fn, get_train_transform, get_valid_transform
from sklearn.model_selection import train_test_split

# The dataset class.
class CustomDataset(Dataset):
    def __init__(self, dir_path, width, height, classes, transforms=None, all_image_paths=None, all_images=None):
        self.transforms = transforms
        self.dir_path = dir_path
        self.height = height
        self.width = width
        self.classes = classes
        self.image_file_types = ['*.jpg', '*.jpeg', '*.png', '*.ppm', '*.JPG']
        self.all_image_paths = all_image_paths if all_image_paths is not None else []
        self.all_images = all_images if all_images is not None else []
        
        # Get all the image paths in sorted order (only if not provided as arguments).
        if self.dir_path is not None and len(self.all_image_paths) == 0:
            for file_type in self.image_file_types:
                self.all_image_paths.extend(glob.glob(os.path.join(self.dir_path, file_type)))
            self.all_images = [image_path.split(os.path.sep)[-1] for image_path in self.all_image_paths]
            self.all_images = sorted(self.all_images)

    def __getitem__(self, idx):
        # Capture the image name and the full image path.
        image_name = self.all_images[idx]
        image_path = os.path.join(self.dir_path, image_name)

        # Read and preprocess the image.
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_resized = cv2.resize(image, (self.width, self.height))
        image_resized /= 255.0
        
        # Capture the corresponding XML file for getting the annotations.
        annot_filename = os.path.splitext(image_name)[0] + '.xml'
        annot_file_path = os.path.join(self.dir_path, annot_filename)
        
        boxes = []
        labels = []
        tree = et.parse(annot_file_path)
        root = tree.getroot()
        
        # Original image width and height.
        image_width = image.shape[1]
        image_height = image.shape[0]
        
        # Box coordinates for xml files are extracted 
        # and corrected for image size given.
        for member in root.findall('object'):
            # Get label and map the `classes`.
            labels.append(self.classes.index(member.find('name').text))
            
            # Left corner x-coordinates.
            xmin = int(member.find('bndbox').find('xmin').text)
            # Right corner x-coordinates.
            xmax = int(member.find('bndbox').find('xmax').text)
            # Left corner y-coordinates.
            ymin = int(member.find('bndbox').find('ymin').text)
            # Right corner y-coordinates.
            ymax = int(member.find('bndbox').find('ymax').text)
            
            # Resize the bounding boxes according 
            # to resized image `width`, `height`.
            xmin_final = (xmin/image_width)*self.width
            xmax_final = (xmax/image_width)*self.width
            ymin_final = (ymin/image_height)*self.height
            ymax_final = (ymax/image_height)*self.height

            # Check that max coordinates are at least one pixel
            # larger than min coordinates.
            if xmax_final == xmin_final:
                xmax_final += 1
            if ymax_final == ymin_final:
                ymax_final += 1
            # Check that all coordinates are within the image.
            if xmax_final > self.width:
                xmax_final = self.width
            if ymax_final > self.height:
                ymax_final = self.height
            
            boxes.append([xmin_final, ymin_final, xmax_final, ymax_final])
        
        # Bounding box to tensor.
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Area of the bounding boxes.
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if len(boxes) > 0 \
            else torch.as_tensor(boxes, dtype=torch.float32)
        # No crowd instances.
        iscrowd = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        # Labels to tensor.
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # Prepare the final `target` dictionary.
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
        image_id = torch.tensor([idx])
        target["image_id"] = image_id

        # Apply the image transforms.
        if self.transforms:
            sample = self.transforms(image = image_resized,
                                     bboxes = target['boxes'],
                                     labels = labels)
            image_resized = sample['image']
            target['boxes'] = torch.Tensor(sample['bboxes'])
        
        if np.isnan((target['boxes']).numpy()).any() or target['boxes'].shape == torch.Size([0]):
            target['boxes'] = torch.zeros((0, 4), dtype=torch.int64)
        return image_resized, target

    def __len__(self):
        return len(self.all_images)

# Prepare the final datasets and data loaders.
def create_trainval_dataset(DIR):
    trainval_dataset = CustomDataset(
        DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
    )
    return trainval_dataset

# Prepare the fold datasets using all_image_paths and all_images from the `trainval_dataset` object.
def create_train_dataset(all_image_paths, all_images):
    train_dataset = CustomDataset(
        None, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform(), all_image_paths=all_image_paths, all_images=all_images
    )
    return train_dataset
def create_valid_dataset(all_image_paths, all_images):
    valid_dataset = CustomDataset(
        None, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform(), all_image_paths=all_image_paths, all_images=all_images
    )
# def create_train_dataset(DIR):
#     train_dataset = CustomDataset(
#         DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_train_transform()
#     )
#     return train_dataset
# def create_valid_dataset(DIR):
#     valid_dataset = CustomDataset(
#         DIR, RESIZE_TO, RESIZE_TO, CLASSES, get_valid_transform()
#     )
    return valid_dataset
def create_train_loader(train_dataset, num_workers=0):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return train_loader
def create_valid_loader(valid_dataset, num_workers=0):
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        drop_last=True
    )
    return valid_loader


# execute `datasets.py`` using Python command from 
# Terminal to visualize sample images
# USAGE: python datasets.py
if __name__ == '__main__':
    # sanity check of the Dataset pipeline with sample visualization
    # dataset = CustomDataset(
    #     TRAIN_DIR, RESIZE_TO, RESIZE_TO, CLASSES
    # )
    # print(f"Number of training images: {len(dataset)}")
    
    # function to visualize a single sample
    # def visualize_sample(image, target):
    #     for box_num in range(len(target['boxes'])):
    #         box = target['boxes'][box_num]
    #         label = CLASSES[target['labels'][box_num]]
    #         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    #         cv2.rectangle(
    #             image, 
    #             (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
    #             (0, 0, 255), 
    #             2
    #         )
    #         cv2.putText(
    #             image, 
    #             label, 
    #             (int(box[0]), int(box[1]-5)), 
    #             cv2.FONT_HERSHEY_SIMPLEX, 
    #             0.7, 
    #             (0, 0, 255), 
    #             2
    #         )
    #     cv2.imshow('Image', image)
    #     cv2.waitKey(0)
        
    # NUM_SAMPLES_TO_VISUALIZE = 5
    # for i in range(NUM_SAMPLES_TO_VISUALIZE):
    #     image, target = dataset[i]
    #     visualize_sample(image, target)

    # Split dataset into train and test sets
    dataset_dir = Path(r'D:\Patrick_PAST_UNI_STUFF\Honours\Handfish Detections\Handfish Detections - Human - Tight Boxes')
    project_root = Path(__file__).resolve().parents[1]
    train_path = project_root / 'retina_net' / 'data' / 'trainval'
    test_path = project_root / 'retina_net' / 'data' / 'test'

    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    image_file_names = []
    for file in dataset_dir.iterdir():
        if file.is_file() and file.suffix.lower() in {'.jpg', '.jpeg', '.png', '.ppm'}:
            xml_name = file.with_suffix('.xml').name
            # keep only valid image/xml pairs
            if (dataset_dir / xml_name).exists():
                image_file_names.append(file.name)

    if len(image_file_names) == 0:
        raise RuntimeError(f"No valid image/xml pairs found in: {dataset_dir}")

    x_trainval, x_test = train_test_split(
        image_file_names,
        test_size=0.1,
        random_state=42,
        shuffle=True
    )

    os.makedirs(train_path, exist_ok=True)
    os.makedirs(test_path, exist_ok=True)
    for file in x_trainval:
        print(f"Processing train file: {file}")
        xml_file = file[:-3] + 'xml'
        shutil.copy2(dataset_dir / file, train_path / file)
        shutil.copy2(dataset_dir / xml_file, train_path / xml_file)
    for file in x_test:
        print(f"Processing test file: {file}")
        xml_file = file[:-3] + 'xml'
        shutil.copy2(dataset_dir / file, test_path / file)
        shutil.copy2(dataset_dir / xml_file, test_path / xml_file)

    print(f"Copied {len(x_trainval)} image/xml pairs to: {train_path}")
    print(f"Copied {len(x_test)} image/xml pairs to: {test_path}")
    
    
