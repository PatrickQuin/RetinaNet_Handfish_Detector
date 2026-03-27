import os
import sys

from pathlib import Path
import numpy as np 

# Allow running this file directly: `python retina_net/train.py`
if __package__ is None or __package__ == '':
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from retina_net.config import (
    DEVICE, 
    NUM_CLASSES, 
    NUM_EPOCHS, 
    NUM_FOLDS,
    OUT_DIR,
    VISUALIZE_TRANSFORMED_IMAGES, 
    NUM_WORKERS
)

from retina_net.model import create_model
from retina_net.custom_utils import (
    Averager, 
    SaveBestModel, 
    save_model, 
    save_loss_plot,
    save_mAP
)
from tqdm.auto import tqdm
from retina_net.datasets import (
    create_trainval_dataset,
    create_train_dataset, 
    create_valid_dataset, 
    create_train_loader, 
    create_valid_loader
)
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torch.optim.lr_scheduler import StepLR

import torch
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import train_test_split, KFold


plt.style.use('ggplot')

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed) 

# Function for running training iterations.
def train(train_data_loader, model):
    print('Training')
    model.train()
    
     # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    
    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()
    
        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return loss_value

# Function for running validation iterations.
def validate(valid_data_loader, model):
    print('Validating')
    model.eval()
    
    # Initialize tqdm progress bar. 
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))
    target = []
    preds = []
    for i, data in enumerate(prog_bar):
        images, targets = data
        
        images = list(image.to(DEVICE) for image in images)
        targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]
        
        with torch.no_grad():
            outputs = model(images, targets)

        # For mAP calculation using Torchmetrics.
        #####################################
        for i in range(len(images)):
            true_dict = dict()
            preds_dict = dict()
            true_dict['boxes'] = targets[i]['boxes'].detach().cpu()
            true_dict['labels'] = targets[i]['labels'].detach().cpu()
            preds_dict['boxes'] = outputs[i]['boxes'].detach().cpu()
            preds_dict['scores'] = outputs[i]['scores'].detach().cpu()
            preds_dict['labels'] = outputs[i]['labels'].detach().cpu()
            preds.append(preds_dict)
            target.append(true_dict)
        #####################################

    metric.reset()
    metric.update(preds, target)
    metric_summary = metric.compute()
    return metric_summary

if __name__ == '__main__':
    os.makedirs('outputs', exist_ok=True)
    # Initialize the model and move to the computation device.
    model = create_model(num_classes=NUM_CLASSES)
    model = model.to(DEVICE)
    print(model)
    # Total parameters and trainable parameters.
    total_params = sum(p.numel() for p in model.parameters())
    print(f"{total_params:,} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{total_trainable_params:,} training parameters.")
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.001, momentum=0.9, nesterov=True)
    scheduler = StepLR(
        optimizer=optimizer, step_size=10, gamma=0.1, verbose=True
    )

    # To monitor training loss
    train_loss_hist = Averager()
    # To store training loss and mAP values.
    train_loss_list = []
    map_50_list = []
    map_list = []

    # Name to save the trained model with.
    MODEL_NAME = 'model'

    # Path to data
    project_root = Path(__file__).resolve().parents[1]
    trainval_path = project_root / 'retina_net' / 'data' / 'trainval'

    trainval_dataset = create_trainval_dataset(trainval_path)

    # To save best model.
    save_best_model = SaveBestModel()

    metric = MeanAveragePrecision()

    # Define number of splits for K-Fold Cross Validation.
    kf = KFold(n_splits=NUM_FOLDS, shuffle=True, random_state=seed)

    fold = 1
    for train_index, val_index in kf.split(trainval_dataset):
        print(f"\nSTARTING NEXT FOLD: FOLD {fold} of {NUM_FOLDS}")

        # Create the datasets and data loaders using the test val splits
        # Convert numpy array indices to lists for Python list indexing
        train_index = train_index
        val_index = val_index
        
        train_dataset_images = np.array(trainval_dataset.all_image_paths)[train_index].tolist()
        train_dataset_paths  = np.array(trainval_dataset.all_image_paths[train_index]).tolist()
        val_dataset_images = np.array(trainval_dataset.all_image_paths[val_index]).tolist()
        val_dataset_paths  = np.array(trainval_dataset.all_image_paths[val_index]).tolist()

        train_dataset = create_train_dataset(train_dataset_paths, train_dataset_images)
        val_dataset = create_valid_dataset(val_dataset_paths, val_dataset_images)

        train_loader = create_train_loader(train_dataset, NUM_WORKERS)
        valid_loader = create_valid_loader(val_dataset, NUM_WORKERS)

        print(f"Number of training samples: {len(train_dataset)}")
        print(f"Number of validation samples: {len(val_dataset)}\n")

        # Whether to show transformed images from data loader or not.
        if VISUALIZE_TRANSFORMED_IMAGES:
            from retina_net.custom_utils import show_tranformed_image
            show_tranformed_image(train_loader)

        # Training loop.
        for epoch in range(NUM_EPOCHS):
            print(f"\nEPOCH {fold} of {NUM_FOLDS}")
            print(f"\nEPOCH {epoch+1} of {NUM_EPOCHS}")

            # Reset the training loss histories for the current epoch.
            train_loss_hist.reset()

            # Start timer and carry out training and validation.
            start = time.time()
            train_loss = train(train_loader, model)
            metric_summary = validate(valid_loader, model)
            print(f"Epoch #{epoch+1} train loss: {train_loss_hist.value:.3f}")   
            print(f"Epoch #{epoch+1} mAP: {metric_summary['map']}") 
            print(f"Best mAP: {metric_summary['map_50']}")  
            end = time.time()
            print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

            train_loss_list.append(train_loss)
            map_50_list.append(metric_summary['map_50'])
            map_list.append(metric_summary['map'])

            print(f'Saving best model to {OUT_DIR}')
            # save the best model till now.
            save_best_model(
                model, float(metric_summary['map']), epoch, 'outputs'
            )
            # Save the current epoch model.
            save_model(epoch, model, optimizer)

            # Save loss plot.
            save_loss_plot(OUT_DIR, train_loss_list)

            # Save mAP plot.
            save_mAP(OUT_DIR, map_50_list, map_list)
            # scheduler.step()
        fold += 1
