# RetinaNet Handfish Detector
When opening the project, make sure that the RetinaNet_Handfish_Detector is set as the root. If this is not done there will be directory errors. Once open the required libraries can be downlowded using the requirements file.
## requirements.txt
In the terminal run the following command to read the requirements.txt and download all the required libraries.
```bash
pip install -r requirements.txt
```
## retina_net
### *config.py*
This is the file where configurations for the program can be easily controlled without needing to change a bunch of instances (with a few exceptions)
Some training and processing hyper parameters can be controlled here:
- Batch size
- Image size
- Number of Folds
- Number of Epochs per fold
- Number of Workers

The classes can be controlled here. Say you wanted to differentiate between **Red** and **Spotted** handfish you could add those classes here, **but**, you would also need to update the annotation xml files.

### *custom_utils.py*
This file contains untility functions to apply transforms, save models and generate and save training plots. 

### *datasets.py*
In this file you can create the train-test split. It also prepares the dataset for use in training and testing.

### *model.py*
This is where you can choose and design the model to retrain

### *train.py*
This is where the training of the model is run from.

### *eval.py*
Model Evaluation after testing. Set to test the best model from the training run

### *inference.py*
This can be used to process images and get an output with a prediction and bounding box. How the box is draw, colours, text ect can be varied in here.

## **Execution Order**
The Files should be run in the order:
1. dataset.py
2. train.py
3. eval.py/inference.py

## *data_augmentations*
This folder contains 2 files.
- *col_to_gr.py*
- *std_ratio.py*

These augmentations are applied directly to the images and not the ones loaded into the environment. So be careful not to overwrite images you want to keep. 
*col_to_gr.py* creates images containing only the r,g,b channels select by adjusting the code. This was used to try and understand which channels held the most weight to experiment if strengthening a channel could improve results.

*std_ratio.py* performs a colour correction that uses the ratio of the standard deviation between the dominant and weakest fields.

## yolo
The YOLO model was what I started with before going away from it and then as a last minute attempt tried it again. It simple, frankly bad, but fast.  The Annotations need to be reformatted to suit YOLO format. It also requires very specific folder structure which although missing image data, is already made.

## old
These are files that I did use but have somewhat made redundant. Look at them if you like but probably not much use.


