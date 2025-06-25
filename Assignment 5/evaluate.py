import pickle
import yaml
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, confusion_matrix

# Load experiment parameters from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Load the test dataset from the pickle file
with open("test.pkl", "rb") as f:
    test_data = pickle.load(f)

# Convert images to tensors, normalize pixel values to [0,1], and adjust shape for PyTorch
test_images = torch.tensor(
    np.stack([entry["image"] for entry in test_data]), dtype=torch.float32
).permute(0, 3, 1, 2) / 255.0  # Change from (H, W, C) to (C, H, W)

# Convert labels to tensor
test_labels = torch.tensor([entry["label"] for entry in test_data], dtype=torch.long)

# Wrap images and labels in a DataLoader for batch processing
test_loader = DataLoader(
    TensorDataset(test_images, test_labels),
    batch_size=params["testing"]["batch_size"],
    shuffle=False  # No need to shuffle test data
)

# Load the trained model from the pickle file
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Set model to evaluation mode (disables dropout, batch norm updates, etc.)
model.eval()

# Lists to store predictions and true labels
all_preds, all_labels = [], []

# Disable gradient computation (saves memory and speeds up inference)
with torch.no_grad():
    for images, labels in test_loader:
        outputs = model(images)  # Get model predictions
        all_preds.extend(outputs.argmax(1).numpy())  # Get predicted class indices
        all_labels.extend(labels.numpy())  # Store actual labels

# Compute overall test accuracy
accuracy = accuracy_score(all_labels, all_preds)

# Generate confusion matrix for class-wise evaluation
conf_matrix = confusion_matrix(all_labels, all_preds).tolist()  # Convert to list for JSON storage

# Save evaluation results (accuracy and confusion matrix) to a JSON file
evaluation_results = {
    "test_accuracy": accuracy,
    "confusion_matrix": conf_matrix
}

with open("evaluation_report.json", "w") as f:
    json.dump(evaluation_results, f)
