import pickle
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from itertools import product
import json

# Load experiment parameters from params.yaml
with open("params.yaml") as f:
    params = yaml.safe_load(f)

# Set random seeds for reproducibility
torch.manual_seed(params["random_seed"])
np.random.seed(params["random_seed"])


def load_data(file):
    """Loads dataset from a pickle file and converts it into a PyTorch TensorDataset."""
    with open(file, "rb") as f:
        data = pickle.load(f)

    # Convert image data to tensors and normalize pixel values to [0,1]
    images = torch.tensor(np.stack([d["image"] for d in data]), dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    labels = torch.tensor([d["label"] for d in data], dtype=torch.long)

    return torch.utils.data.TensorDataset(images, labels)


# Load training and validation datasets into DataLoaders
train_loader = torch.utils.data.DataLoader(
    load_data("train.pkl"),
    batch_size=params["training"]["batch_size"],
    shuffle=True
)
val_loader = torch.utils.data.DataLoader(
    load_data("val.pkl"),
    batch_size=params["training"]["batch_size"],
    shuffle=False
)


def build_model(conv_layers, filters, kernel_size, maxpool_kernel, maxpool_stride):
    """Builds a CNN model dynamically based on hyperparameters."""
    layers = []
    in_channels = 3  # Input has 3 color channels (RGB)

    # Create specified number of convolutional layers
    for _ in range(conv_layers):
        layers += [
            nn.Conv2d(in_channels, filters, kernel_size, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(maxpool_kernel, maxpool_stride)
        ]
        in_channels = filters  # Update input channels for the next conv layer

    # Wrap conv layers in a sequential model
    conv = nn.Sequential(*layers)

    # Determine the input size of the fully connected layer
    fc_input_size = torch.numel(conv(torch.zeros(1, 3, 32, 32)).flatten(1))

    # Return full model: CNN feature extractor + classifier
    return nn.Sequential(
        conv,
        nn.Flatten(),
        nn.Linear(fc_input_size, 10)  # Output layer for 10 classes
    )


# Track the best validation accuracy and corresponding hyperparameters
best_acc, best_params = 0, {}

# Perform grid search over the hyperparameter combinations
for conv_layers, filters in product(
    params["training"]["tuning"]["conv_layers"],
    params["training"]["tuning"]["conv_filters"]
):
    # Build model with the current hyperparameter combination
    model = build_model(
        conv_layers,
        filters,
        params["training"]["kernel_size"],
        params["training"]["maxpool_kernel_size"],
        params["training"]["maxpool_stride"]
    )

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=params["training"]["lr"])
    criterion = nn.CrossEntropyLoss()

    # Train the model for the specified number of epochs
    for _ in range(params["training"]["epochs"]):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(images), labels)
            loss.backward()
            optimizer.step()

    # Evaluate model on the validation set
    model.eval()
    correct = sum((torch.argmax(model(images), 1) == labels).sum().item() for images, labels in val_loader)
    acc = correct / len(val_loader.dataset)  # Compute validation accuracy

    # Save model if it has the highest validation accuracy so far
    if acc > best_acc:
        best_acc, best_params = acc, {"conv_layers": conv_layers, "filters": filters}
        with open("model.pkl", "wb") as f:
            pickle.dump(model, f)

# Save the best hyperparameter configuration and validation accuracy
with open("model_params.json", "w") as f:
    json.dump({"chosen_hyperparameters": best_params, "validation_accuracy": best_acc}, f)
