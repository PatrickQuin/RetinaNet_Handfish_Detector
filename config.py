import torch

BATCH_SIZE = 4 # Increase / decrease according to GPU memeory.
RESIZE_TO = 640 # Resize the image for training and transforms.
NUM_EPOCHS = 25 # Number of epochs to train for.
NUM_WORKERS = 4 # Number of parallel workers for data loading.
NUM_FOLDS = 5 # Number of folds for cross-validation.

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Training images and XML files directory.
TRAIN_DIR = 'data/train'
# Validation images and XML files directory.
VALID_DIR = 'data/val'

TEST_DIR = 'data/test'
# Classes: 0 index is reserved for background.
CLASSES = [
    '__background__', 'Brachionichthyidae'
]

NUM_CLASSES = len(CLASSES)

# Whether to visualize images after crearing the data loaders.
VISUALIZE_TRANSFORMED_IMAGES = False

# Location to save model and plots.
OUT_DIR = 'outputs'