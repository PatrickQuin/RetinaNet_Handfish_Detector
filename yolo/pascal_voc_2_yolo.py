import os
import xml.etree.ElementTree as ET

# Configuration
input_dir = 'C:/Users/pmqui/OneDrive - University of Tasmania/Honours/Repositories/Handfish Detections - Human - Tight Boxes'     # Directory with XML files
output_dir = 'yolo_data/training'        # Directory to save YOLO txt files
class_list = ['Brachionichthyidae']       # Add more classes if needed

os.makedirs(output_dir, exist_ok=True)

def convert_bbox(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    w = box[2] - box[0]
    h = box[3] - box[1]
    return x_center * dw, y_center * dh, w * dw, h * dh

for xml_file in os.listdir(input_dir):
    if not xml_file.endswith('.xml'):
        continue

    tree = ET.parse(os.path.join(input_dir, xml_file))
    root = tree.getroot()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name not in class_list:
            continue  # Skip unknown classes
        class_id = class_list.index(class_name)

        bndbox = obj.find('bndbox')
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        xmax = int(bndbox.find('xmax').text)
        ymax = int(bndbox.find('ymax').text)

        bbox = convert_bbox((width, height), (xmin, ymin, xmax, ymax))
        yolo_line = f"{class_id} {' '.join(f'{coord:.6f}' for coord in bbox)}"
        yolo_lines.append(yolo_line)

    txt_filename = os.path.splitext(xml_file)[0] + '.txt'
    with open(os.path.join(output_dir, txt_filename), 'w') as out_file:
        out_file.write('/n'.join(yolo_lines))
