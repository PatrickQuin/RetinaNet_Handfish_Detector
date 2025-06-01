import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import glob as glob
import time
from ultralytics import YOLO
from xml.etree import ElementTree as ET

from model import create_model

CLASSES = [
    '__background__', 'Brachionichthyidae'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = [(0, 0, 255) for _ in CLASSES]  # BGR for red in OpenCV

def get_detections(image_path, model, threshold=0.25, imgsz=640):
    """
    Perform inference using a YOLO model and return detections.
    """
    # Perform inference
    results = model.predict(source=image_path, conf=threshold, imgsz=imgsz, device=DEVICE, verbose=False)

    # Extract detections
    detections = []
    for result in results:
        boxes = result.boxes.xyxy.cpu().numpy()  # Bounding boxes (x_min, y_min, x_max, y_max)
        scores = result.boxes.conf.cpu().numpy()  # Confidence scores
        labels = result.boxes.cls.cpu().numpy().astype(int)  # Class labels

        for i, box in enumerate(boxes):
            if scores[i] >= threshold:
                detections.append(box.tolist())  # Append bounding box

    return detections


def parse_voc_annotation(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    boxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        boxes.append([xmin, ymin, xmax, ymax])
    return boxes

def iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    return interArea / float(boxAArea + boxBArea - interArea)

def evaluate_detector(directory, model, iou_threshold=0.0):
    from collections import Counter

    frame_count = 0 # To count total frames.
    total_fps = 0 # To get the final frames per second.

    y_true = []
    y_pred = []
    count = 1
    for filename in os.listdir(directory):

        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(directory, filename)
        base_name = os.path.splitext(filename)[0]
        xml_path = os.path.join(directory, base_name + ".xml")
        start_time = time.time()
        detections = get_detections(image_path, model)  # List of [xmin, ymin, xmax, ymax]
        end_time = time.time()
        fps = 1 / (end_time - start_time)
        # Total FPS till current frame.
        total_fps += fps
        frame_count += 1

        has_annotation = os.path.exists(xml_path)

        if has_annotation:
            gt_boxes = parse_voc_annotation(xml_path)
            match_found = False

            # Check if any detection matches any GT (only count one)
            for det in detections:
                for gt in gt_boxes:
                    if iou(det, gt) > iou_threshold:
                        match_found = True
                        break
                if match_found:
                    break

            y_true.append(1)  # Ground truth: object is present
            y_pred.append(1 if match_found else 0)

        else:
            # No object is present
            y_true.append(0)
            y_pred.append(1 if detections else 0)
            # Get the current fps.
        print(f"Image: {count}" )
        count += 1

    # Confusion matrix
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Handfish", "Handfish"])
    disp.plot(cmap="Reds")
    plt.title("Per-Image Confusion Matrix")
    plt.savefig("confusion_matrix_image_level.png", dpi=350)
    plt.close()
    print("Confusion matrix saved as 'confusion_matrix_image_level.png'")
    return cm

if __name__ == "__main__":
    model = create_model(num_classes=NUM_CLASSES)
    checkpoint = torch.load('TH_Yolo/outputs/best_model.pt', map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(DEVICE).eval()
    # Set to the folder with images and XML annotations
    directory = "images/"

    cm = evaluate_detector(directory, model=model)
    print("Confusion Matrix:\n", cm)
