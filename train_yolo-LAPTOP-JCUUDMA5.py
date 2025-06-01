import os
import pandas as pd
import yaml
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt
from ultralytics import YOLO

# Paths to the YAML files for each fold
fold_yaml_files = [
    "data0.yaml",
    "data1.yaml",
    "data2.yaml"
]

# Path to the YOLO model weights
weights_path = "yolov8s.pt"

# Training parameters
batch = 16
imgsz = 1024
project = "YOLO_HANDFISH_DETECTOR_TRAINING"
epochs = 50

# Directory to save outputs
outputs_dir = "outputs"
os.makedirs(outputs_dir, exist_ok=True)

# Load the YOLO model
model = YOLO(weights_path, task="detect")

# Dictionary to store results for each fold
# Train on each fold
for fold, yaml_file in enumerate(fold_yaml_files):
    print(f"Training on Fold {fold} using {yaml_file}...")
    
    # Train the model using the current fold's YAML file
    model.train(
        data=yaml_file,  # Path to the current fold's dataset YAML
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        project=project,
        name=f"fold_{fold}"  # Name for the training run
    )
