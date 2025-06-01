import numpy as np
import cv2
import torch
import glob
import os
import time
import argparse
from xml.etree import ElementTree as et
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from model import create_model

# Constants
CLASSES = ['__background__', 'Brachionichthyidae']
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
NUM_THREADS = 8

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', help='path to input image directory')
parser.add_argument('--imgsz', default=None, type=int, help='image resize shape')
parser.add_argument('--threshold', default=0.25, type=float, help='detection threshold')
args = vars(parser.parse_args())

# Directories
DIR_TEST = args['input']
os.makedirs('inference_outputs/images', exist_ok=True)

# Model setup
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# List of images
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Total test images: {len(test_images)}")

# Queue of images to process
image_queue = Queue()
for img_path in test_images:
    image_queue.put(img_path)

def process_image():
    while not image_queue.empty():
        try:
            image_path = image_queue.get_nowait()
        except:
            break

        image_name = os.path.basename(image_path).split('.')[0]
        ann_path = os.path.join(DIR_TEST, image_name + '.xml')

        if not os.path.exists(ann_path):
            print(f"Annotation for {image_name} not found, skipping.")
            return

        image = cv2.imread(image_path)
        orig_image = image.copy()

        if args['imgsz'] is not None:
            image = cv2.resize(image, (args['imgsz'], args['imgsz']))

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image_rgb /= 255.0
        image_input = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
        image_input = torch.tensor(image_input, dtype=torch.float).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            outputs = model(image_input)

        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]

        if len(outputs[0]['boxes']) == 0:
            return

        boxes = outputs[0]['boxes'].numpy()
        scores = outputs[0]['scores'].numpy()
        labels = outputs[0]['labels'].numpy()

        boxes = boxes[scores >= args['threshold']].astype(np.int32)
        if boxes.shape[0] == 0:
            return

        pred_classes = [CLASSES[i] for i in labels[scores >= args['threshold']]]

        for j, box in enumerate(boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]

            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])

            cv2.rectangle(orig_image, (xmin, ymin), (xmax, ymax), color[::-1], 3)
            cv2.putText(orig_image, class_name, (xmin, ymin - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color[::-1], 2, cv2.LINE_AA)

        out_path = os.path.join('inference_outputs/images', f"{image_name}.jpg")
        cv2.imwrite(out_path, orig_image)
        print(f"Saved detection: {image_name}.jpg")

# Run multithreaded processing
start = time.time()
with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    for _ in range(NUM_THREADS):
        executor.submit(process_image)
end = time.time()

print(f"Processing complete in {end - start:.2f} seconds")
