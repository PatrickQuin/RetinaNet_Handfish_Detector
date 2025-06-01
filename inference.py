import numpy as np
import cv2
import torch
import glob as glob
import os
import time
import argparse
from xml.etree import ElementTree as et

from model import create_model

CLASSES = [
    '__background__', 'Brachionichthyidae'
]
NUM_CLASSES = len(CLASSES)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


np.random.seed(42)

# Construct the argument parser.
parser = argparse.ArgumentParser()
parser.add_argument(
    '-i', '--input', 
    help='path to input image directory',
)
parser.add_argument(
    '--imgsz', 
    default=None,
    type=int,
    help='image resize shape'
)
parser.add_argument(
    '--threshold',
    default=0.25,
    type=float,
    help='detection threshold'
)
args = vars(parser.parse_args())

os.makedirs('inference_outputs/images', exist_ok=True)

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# Load the best model and trained weights.
model = create_model(num_classes=NUM_CLASSES)
checkpoint = torch.load('outputs/best_model.pth', map_location=DEVICE)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE).eval()

# Directory where all the images are present.
DIR_TEST = 'data/test/'#args['input']
test_images = glob.glob(f"{DIR_TEST}/*.jpg")
print(f"Test instances: {len(test_images)}")

frame_count = 0 # To count total frames.
total_fps = 0 # To get the final frames per second.

for i in range(len(test_images)):
    # Get the image file name for saving output later on.
    image_name = test_images[i].split(os.path.sep)[-1].split('.')[0]
    image = test_images[i]
    #ann_path = os.path.join(DIR_TEST, image_name + '.xml')
    boxes = []
    #labels = []
    #tree = et.parse(ann_path)
    #root = tree.getroot()
    image = cv2.imread(image)
    orig_image = image.copy()
    if args['imgsz'] is not None:
        image = cv2.resize(image, (args['imgsz'], args['imgsz']))
    print(image.shape)
    # BGR to RGB.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    # Make the pixel range between 0 and 1.
    image /= 255.0
    # Bring color channels to front (H, W, C) => (C, H, W).
    image_input = np.transpose(image, (2, 0, 1)).astype(np.float32)
    # Convert to tensor.
    image_input = torch.tensor(image_input, dtype=torch.float).cpu()
    # Add batch dimension.
    image_input = torch.unsqueeze(image_input, 0)
    start_time = time.time()
    # Predictions
    with torch.no_grad():
        outputs = model(image_input.to(DEVICE))
    end_time = time.time()

    # Get the current fps.
    #fps = 1 / (end_time - start_time)
    # Total FPS till current frame.
    #total_fps += fps
    #frame_count += 1

    # Load all detection to CPU for further operations.
    outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
    # Carry further only if there are detected boxes.
    if len(outputs[0]['boxes']) != 0:
        boxes = outputs[0]['boxes'].data.numpy()
        scores = outputs[0]['scores'].data.numpy()
        # Filter out boxes according to `detection_threshold`.
        boxes = boxes[scores >= args['threshold']].astype(np.int32)
        draw_boxes = boxes.copy()
        # Get all the predicited class names.
        pred_classes = [CLASSES[i] for i in outputs[0]['labels'].cpu().numpy()]
        
        # Draw the bounding boxes and write the class name on top of it.
        for j, box in enumerate(draw_boxes):
            class_name = pred_classes[j]
            color = COLORS[CLASSES.index(class_name)]
            # Rescale boxes.
            xmin = int((box[0] / image.shape[1]) * orig_image.shape[1])
            ymin = int((box[1] / image.shape[0]) * orig_image.shape[0])
            xmax = int((box[2] / image.shape[1]) * orig_image.shape[1])
            ymax = int((box[3] / image.shape[0]) * orig_image.shape[0])
            cv2.rectangle(orig_image,
                        (xmin, ymin),
                        (xmax, ymax),
                        color[::-1], 
                        3)
            cv2.putText(orig_image, 
                        class_name, 
                        (xmin, ymin-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        color[::-1], 
                        2, 
                        lineType=cv2.LINE_AA)
        # check if the image is annotated
        # if len(root.findall('object')) != 0:
        #     # Draw the ground truth boxes.
        #     for member in root.findall('object'):
        #         # Left corner x-coordinates.
        #         xmin = int(member.find('bndbox').find('xmin').text)
        #         # Right corner x-coordinates.
        #         xmax = int(member.find('bndbox').find('xmax').text)
        #         # Left corner y-coordinates.
        #         ymin = int(member.find('bndbox').find('ymin').text)
        #         # Right corner y-coordinates.
        #         ymax = int(member.find('bndbox').find('ymax').text)

        #         cv2.rectangle(
        #             orig_image, 
        #             (xmin, ymin), (xmax, ymax),
        #             (0, 255, 0), 
        #             2
        #         )
        #cv2.imshow('Prediction', orig_image)
        #cv2.waitKey(1)
        cv2.imwrite(f"inference_outputs/images/{image_name}.jpg", orig_image)
    print(f"Image {i+1} done...")
    print('-'*50)


print('TEST PREDICTIONS COMPLETE')
#cv2.destroyAllWindows()
# Calculate and print the average FPS.
#avg_fps = total_fps / frame_count
#print(f"Average FPS: {avg_fps:.3f}")