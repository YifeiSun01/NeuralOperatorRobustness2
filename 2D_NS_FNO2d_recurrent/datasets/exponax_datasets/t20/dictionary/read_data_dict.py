from pathlib import Path
import torch

# Get the directory containing this script
script_dir = Path(__file__).parent.resolve()

# Load and analyze main dataset
print("=== Loading Main Dataset ===")
dataset_path = script_dir / 'dim2d_nx256_N2000_solver=exponax_nu0.000_t20.0_dict_ntimepoints21_batch0_all_frames.pt'
dataset = torch.load(dataset_path, weights_only=False)
print("Dataset keys available:", dataset.keys())
print("Input data (x) shape - [samples, height, width]:", dataset["x"].shape)
print("Output data (y) shape - [samples, height, width, timesteps]:", dataset["y"].shape)

# Load and analyze test dataset
print("\n=== Loading Test Dataset ===")
test_dataset_path = script_dir.parent / 'dim2d_nx256_N50_solver=exponax_nu0.000_t20.0_test_all_frames.pt'
test_dataset = torch.load(test_dataset_path, weights_only=False)
print("Test dataset keys available:", test_dataset.keys())
print("Test input data (x) shape:", test_dataset["x"].shape)
print("Test output data (y) shape:", test_dataset["y"].shape)

# Load and analyze train dataset
print("\n=== Loading Training Dataset ===")
train_dataset_path = script_dir.parent / 'dim2d_nx256_N1150_solver=exponax_nu0.000_t20.0_train_all_frames.pt'
train_dataset = torch.load(train_dataset_path, weights_only=False)
print("Training dataset keys available:", train_dataset.keys())
print("Training input data (x) shape:", train_dataset["x"].shape)
print("Training output data (y) shape:", train_dataset["y"].shape)

# Calculate and print dataset statistics
print("\n=== Dataset Statistics ===")
print(f"Total samples in main dataset: {dataset['x'].shape[0]}")
print(f"Total samples in training set: {train_dataset['x'].shape[0]}")
print(f"Total samples in test set: {test_dataset['x'].shape[0]}")
print(f"Combined train+test samples: {train_dataset['x'].shape[0] + test_dataset['x'].shape[0]}")
print(f"Image dimensions: {dataset['x'].shape[1:]} (height × width)")
print(f"Number of timesteps in output: {dataset['y'].shape[3]}")


from tqdm import tqdm

# Combine train and test x
train_test_x = torch.cat([train_dataset['x'], test_dataset['x']], dim=0)

# Initialize a dictionary to track duplicates
duplicates = {}

# Compare each sample in train_test against all samples in dataset
for i in tqdm(range(len(train_test_x)), desc="Checking train/test samples"):
    current_sample = train_test_x[i]
    
    # Compare with all samples in main dataset with progress bar
    for j in tqdm(range(len(dataset['x'])), desc=f"Comparing sample {i}", leave=False):
        if torch.allclose(current_sample, dataset['x'][j], atol=1e-8):
            duplicates[f"train_test_{i}"] = f"dataset_{j}"
            print(f"\nFound duplicate: train/test sample {i} matches dataset sample {j}")
            break  # Stop checking after first match

if not duplicates:
    print("\nNo identical tensors found between train+test and main dataset")
else:
    print(f"\nFound {len(duplicates)} duplicate pairs:")
    for k, v in duplicates.items():
        print(f"{k} matches {v}")


