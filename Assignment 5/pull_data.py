import os
import pickle
import yaml
import subprocess
from PIL import Image
import numpy as np

# Directory containing dataset partitions
data_dir = "data"

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Get the dataset version(s) to pull from DVC
versions = params["dataset_version"]
versions_to_pull = versions.split()  # Split in case multiple versions are specified

# Lists to store image data and corresponding labels
images, labels = [], []

# Loop through each dataset version specified in params.yaml
for version in versions_to_pull:
    # Checkout the DVC metadata file (ensures correct tracking)
    subprocess.run(["git", "checkout", "--", "data.dvc"], check=True)

    # Checkout the specified dataset version using Git tags
    subprocess.run(["git", "checkout", version], check=True)

    # Pull the dataset files using DVC
    subprocess.run(["dvc", "pull"])

    # Get class names from the dataset directory and assign numerical labels
    class_names = sorted(os.listdir(data_dir))  # Ensure consistent ordering
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}  # Map class names to indices

    # Iterate over each class folder and process images
    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):  # Ensure it's a directory (not a stray file)
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)

                # Open the image, convert it to RGB, and store it as a NumPy array
                img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)

                # Append image data and corresponding label
                images.append(img)
                labels.append(class_to_idx[class_name])

# Combine images and labels into a list of dictionaries
data = [{"image": img, "label": label} for img, label in zip(images, labels)]

# Save the processed dataset as a pickle file
with open("dataset.pkl", "wb") as f:
    pickle.dump(data, f)
