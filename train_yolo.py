import pandas as pd
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

dataset_path = Path("./path/to/dataset")
labels = sorted(dataset_path.rglob("labels/*.txt"))

# Assuming your classes are defined in a YAML file
with open("path/to/data.yaml", "r") as y:
    classes = yaml.safe_load(y)["names"]
cls_idx = sorted(classes.keys())

labels_df = pd.DataFrame([], columns=cls_idx, index=[l.stem for l in labels])

for label in labels:
    lbl_counter = Counter()
    with open(label, "r") as lf:
        lines = lf.readlines()
    for l in lines:
        lbl_counter[int(l.split(" ")[0])] += 1
    labels_df.loc[label.stem] = lbl_counter

labels_df = labels_df.fillna(0.0)

weights_path = "path/to/weights.pt"
model = YOLO(weights_path, task="detect")

results = {}
batch = 16
project = "kfold_demo"
epochs = 50
kfolds = 3

for k, (train_idx, val_idx) in enumerate(kfolds):
    train_files_path = "data/fold{k}/train"
    val_files_path = "data/fold{k}/val"

    # Create dataset YAML for each fold
    dataset_yaml = f"data/fold{k}/dataset.yaml"
    with open(dataset_yaml, "w") as ds_y:
        yaml.safe_dump({
            "path": dataset_path.as_posix(),
            "train": train_files.tolist(),
            "val": val_files.tolist(),
            "names": classes,
        }, ds_y)

    model.train(data=dataset_yaml, epochs=epochs, batch=batch, project=project)
    results[k] = model.metrics