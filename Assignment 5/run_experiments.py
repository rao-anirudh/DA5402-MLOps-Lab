import subprocess
import shutil
import yaml

# Create a backup of the original params.yaml before modifying it
shutil.copy('params.yaml', 'params.yaml.bak')

# Define the different dataset versions and random seeds for experimentation
dataset_versions = ["v1", "v2", "v3", "v1 v2", "v1 v2 v3"]
random_seeds = [5402, 42, 110]

# Load the existing parameters from params.yaml
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)


def update_params(dataset_version, random_seed):
    """Update params.yaml with the specified dataset version and random seed."""

    params['dataset_version'] = dataset_version
    params['random_seed'] = random_seed

    # Save the modified parameters back to params.yaml
    with open('params.yaml', 'w') as f:
        yaml.dump(params, f)


# Iterate over all combinations of dataset versions and random seeds
for version in dataset_versions:
    for seed in random_seeds:
        print(f"Running experiment with dataset version: {version} and random seed: {seed}")

        # Update params.yaml with the current combination
        update_params(version, seed)

        # Run the DVC experiment pipeline with the updated parameters
        subprocess.run(['dvc', 'exp', 'run', '-f'], check=True)

# Save the DVC experiment results in markdown format
subprocess.run(['dvc', 'exp', 'show', '--md'], check=True, stdout=open('dvc_exp_show.txt', 'w'))

# Restore the original params.yaml from the backup
shutil.move('params.yaml.bak', 'params.yaml')

print("Experiment pipeline has been completed for all combinations.")
