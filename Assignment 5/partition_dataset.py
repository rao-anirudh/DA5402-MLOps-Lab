import os
import random
import shutil
import subprocess
import yaml

# Define directories and dataset partitions
base_dir = "cifar10"  # Directory containing the CIFAR-10 dataset
data_dir = "data"  # Directory where partitioned data will be stored
partitions = ["v1", "v2", "v3"]  # Dataset partitions (versions)

# Ensure the data directory exists
os.makedirs(data_dir, exist_ok=True)

# Initialize Git and DVC
subprocess.run(["git", "init"], check=True)  # Initialize a Git repository
subprocess.run(["dvc", "init", "-f"], check=True)  # Initialize a DVC project
subprocess.run(["git", "commit", "-m", "Initialize DVC project"], check=True)  # Commit the initial setup

# Collect all image file paths from the dataset directory
all_images = []
for class_folder in os.listdir(base_dir):
    class_path = os.path.join(base_dir, class_folder)
    if os.path.isdir(class_path):  # Ensure it's a folder (class label)
        images = [os.path.join(class_folder, img) for img in os.listdir(class_path)]
        all_images.extend(images)  # Add images with their class folder prefix

# Shuffle and split the dataset into three partitions (each containing 20,000 images)
random.seed(5402)  # Set a fixed seed for reproducibility
random.shuffle(all_images)  # Shuffle images to ensure randomness
dataset_splits = [all_images[i * 20000:(i + 1) * 20000] for i in range(3)]  # Split into three equal parts

# Process each partition separately
for partition, images in zip(partitions, dataset_splits):
    for img in images:
        src = os.path.join(base_dir, img)  # Source image path
        dest = os.path.join(data_dir, img)  # Destination in data directory
        os.makedirs(os.path.dirname(dest), exist_ok=True)  # Ensure class subdirectories exist
        shutil.copy(src, dest)  # Copy image to its new location

    # Track the dataset partition with DVC
    subprocess.run(["dvc", "add", "data"], check=True)
    subprocess.run(["git", "add", ".gitignore", "data.dvc"], check=True)
    subprocess.run(["git", "commit", "-m", f"Add dataset {partition}"], check=True)

    # Tag this partition version in Git
    subprocess.run(["git", "tag", "-a", partition, "-m", f"Dataset {partition}, 20000 images"], check=True)

    # Clean up the data directory to prepare for the next partition
    for filename in os.listdir(data_dir):
        file_path = os.path.join(data_dir, filename)
        if os.path.isdir(file_path):
            shutil.rmtree(file_path)  # Remove directory if it's a folder
        else:
            os.remove(file_path)  # Remove individual files
