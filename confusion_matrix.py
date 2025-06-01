import os
import xml.etree.ElementTree as ET
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Adjust this function with your detector's output
def get_detections(image_path):
    """
    Dummy detection function.
    Replace this with actual detector inference.
    Returns a list of detected bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    return []  # Replace with real detector output

def parse_voc_annotation(xml_file):
    """
    Parse Pascal VOC XML and return a list of bounding boxes.
    """
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
    """
    Compute Intersection over Union (IoU) between two boxes.
    """
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    if interArea == 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou_value = interArea / float(boxAArea + boxBArea - interArea)
    return iou_value

def evaluate_detector(image_dir, annotation_dir, no_object_images, iou_threshold=0.0):
    y_true = []
    y_pred = []

    for filename in os.listdir(image_dir):
        if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
            continue

        image_path = os.path.join(image_dir, filename)
        detections = get_detections(image_path)
        
        base_name = os.path.splitext(filename)[0]
        annotation_path = os.path.join(annotation_dir, base_name + ".xml")

        if filename in no_object_images:
            # No object should be in the image
            if len(detections) == 0:
                y_true.append(0)  # Negative
                y_pred.append(0)  # TN
            else:
                for _ in detections:
                    y_true.append(0)  # Negative
                    y_pred.append(1)  # FP
        else:
            gt_boxes = parse_voc_annotation(annotation_path)
            matched_gt = set()
            matched_det = set()

            # Match detections with ground truth
            for det_idx, det in enumerate(detections):
                matched = False
                for gt_idx, gt in enumerate(gt_boxes):
                    if iou(det, gt) > iou_threshold:
                        matched_gt.add(gt_idx)
                        matched_det.add(det_idx)
                        matched = True
                        break
                if matched:
                    y_true.append(1)
                    y_pred.append(1)  # TP
                else:
                    y_true.append(0)
                    y_pred.append(1)  # FP

            # False negatives: unmatched ground-truth boxes
            for gt_idx in range(len(gt_boxes)):
                if gt_idx not in matched_gt:
                    y_true.append(1)
                    y_pred.append(0)  # FN

    # Create confusion matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Object", "Object"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    return cm

if __name__ == "__main__":
    image_dir = "/path/to/images"
    annotation_dir = "/path/to/annotations"
    no_object_images = {"image1.jpg", "image2.jpg"}  # set of filenames without the object

    confusion = evaluate_detector(image_dir, annotation_dir, no_object_images)
    print("Confusion Matrix:\n", confusion)