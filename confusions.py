import os
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2
import glob as glob
import time
import argparse
from xml.etree import ElementTree as ET

from model import create_model

CLASSES = [
    '__background__', 'Brachionichthyidae'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = [(0, 0, 255) for _ in CLASSES]  # BGR for red in OpenCV

# Replace this with your actual detection function
def get_detections(image_path, model, threshold=0.25, imgsz=None):

    image = cv2.imread(image_path)
    orig_image = image.copy()
    
    if imgsz is not None:
        image = cv2.resize(image, (imgsz, imgsz))
        
    # BGR to RGB and normalize
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image /= 255.0
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    image_input = torch.tensor(image_input, dtype=torch.float).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(image_input)

    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    
    detections = []
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        labels = outputs[0]['labels'].cpu().numpy()

        boxes = boxes[scores >= threshold].astype(np.int32)
        for i, box in enumerate(boxes):
            detections.append(box.tolist())

            # Optionally draw for debug
            class_name = CLASSES[labels[i]]
            color = COLORS[CLASSES.index(class_name)]
            cv2.rectangle(orig_image,
                          (box[0], box[1]), (box[2], box[3]),
                          color[::-1], 2)
            cv2.putText(orig_image, class_name, 
                        (box[0], box[1]-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                        color[::-1], 2)

    # Optional save/debug
    #os.makedirs('inference_outputs/images', exist_ok=True)
    #image_name = os.path.splitext(os.path.basename(image_path))[0]
    #cv2.imwrite(f'inference_outputs/images/{image_name}.jpg', orig_image)
    print(f'IMAGE DONE {image_path}')
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

    y_true = []
    y_pred = []

    for filename in os.listdir(directory):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        image_path = os.path.join(directory, filename)
        base_name = os.path.splitext(filename)[0]
        xml_path = os.path.join(directory, base_name + ".xml")
        detections = get_detections(image_path, model)  # List of [xmin, ymin, xmax, ymax]

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

    # Confusion matrix
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
