import pickle
import yaml
import numpy as np
from sklearn.model_selection import train_test_split

# Load parameters from params.yaml
with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

# Load the preprocessed dataset from the pickle file
with open("dataset.pkl", "rb") as f:
    dataset = pickle.load(f)

# Extract train, validation, and test split ratios from params.yaml
train_size = params["data_split"]["train"]
val_size = params["data_split"]["val"]
test_size = params["data_split"]["test"]

# First, split the dataset into training data and a temporary set (for validation + test)
train_data, temp_data = train_test_split(
    dataset, train_size=train_size, random_state=params["random_seed"]
)

# Split the temporary set into validation and test data
val_data, test_data = train_test_split(
    temp_data,
    test_size=test_size / (val_size + test_size),  # Adjust proportion for test split
    random_state=params["random_seed"]
)

# Save the split datasets as pickle files
with open("train.pkl", "wb") as f:
    pickle.dump(train_data, f)

with open("val.pkl", "wb") as f:
    pickle.dump(val_data, f)

with open("test.pkl", "wb") as f:
    pickle.dump(test_data, f)
