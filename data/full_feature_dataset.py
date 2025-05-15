import torch
from torch.utils.data import Dataset
import h5py
import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob

def compute_mean_std(feature_folder, basename_list):
    """Computes mean and standard deviation for feature normalization."""
    all_features = []

    # Collect all features from the dataset
    paths = glob.glob(os.path.join(feature_folder, '*.h5'))
    paths = [i for i in paths if os.path.basename(i).split(".h5")[0] in basename_list]

    for file_path in paths:
        with h5py.File(file_path, "r") as f:
            features = f['features'][:]  # Extract features
            all_features.append(features)

    # Stack all features and compute mean & std
    all_features = np.vstack(all_features)
    mean = np.mean(all_features, axis=0)
    std = np.std(all_features, axis=0)

    return mean, std


class FeaturesDataset(Dataset):
    def __init__(
        self,
        feature_folder=None,
        split_csv=None,
        shuffle=True,
        transform=None, 
        mean=None,
        std=None, 
        dataset_type='train'  # Must be 'train', 'val', or 'test'
    ):
        self.feature_folder = feature_folder
        self.shuffle = shuffle
        self.transform = transform
        self.dataset_type = dataset_type
        
        # Load dataset split CSV
        self.split_df = pd.read_csv(split_csv)
        
        # Extract the corresponding file list and labels based on dataset_type
        self.file_list = self.split_df[f'{dataset_type}'].dropna().tolist()
        self.file_labels = self.split_df[f'{dataset_type}_label'].dropna().tolist()

        # Create a dictionary mapping filenames to labels
        self.label_dict = dict(zip(self.file_list, self.file_labels))
        print(self.label_dict)
        
        # Match with available .h5 files
        self.paths = [
            os.path.join(self.feature_folder, f'{basename}.h5')
            for basename in self.file_list
            if os.path.exists(os.path.join(self.feature_folder, f'{basename}.h5'))
        ]
        
        self.indices = np.arange(len(self.paths))
        
        # Store mean and std for normalization
        self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
        self.std = torch.tensor(std, dtype=torch.float32) if std is not None else None 
        
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        """Returns the total number of samples."""
        return len(self.indices)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            dict: {
                'features': Tensor of features,
                'label': Tensor of label,
                'patch_indices': Tensor of patch indices,
                'coordinates': Tensor of coordinates,
                'spixels_indices': Tensor of superpixel indices,
                'file_basename': String file name
            }
        """
        # Get the index of the file
        file_idx = self.indices[index]
        file_path = self.paths[file_idx]
        file_basename = os.path.basename(file_path).split(".")[0]  # Extract filename without extension
        
        # Load the HDF5 file
        with h5py.File(file_path, "r") as f:
            features = f['features'][:]
            patch_indices = f['patch_indices'][:]
            coordinates = f['coordinates'][:]
            spixels_indices = f['spixel_idx'][:]
        
        # Get the label from the CSV
        label = self.label_dict.get(file_basename, 0.0)  # Default to 0.0 if not found
        
        # Convert to tensors
        sample = {
            "features": torch.tensor(features, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.float32).view(1),  # Ensure label is scalar
            "patch_indices": torch.tensor(patch_indices, dtype=torch.int64),
            "coordinates": torch.tensor(coordinates, dtype=torch.float32),
            "spixels_indices": torch.tensor(spixels_indices, dtype=torch.int64),
            "file_basename": file_basename
        }
        
        # Normalize features
        if self.mean is not None and self.std is not None:
            sample["features"] = (sample["features"] - self.mean) / (self.std + 1e-8)  # Avoid division by zero 
        
        # Apply transformation (if any)
        if self.transform:
            sample["features"] = self.transform(sample["features"])
        
        return sample
 
 
# class FeaturesDataset(Dataset):
#     def __init__(
#         self,
#         feature_folder,
#         basename_list=None,
#         feature_file_end ='h5',
#         shuffle=True,
#         transform=None, 
#         mean=None,
#         std=None, 
#         is_train=False, 
#         train_test_val_path=None, 
#         label_path=None, 
#         ):
#         """
#         Args:
#             label_file (str): Path to the CSV file containing labels for each sample.
#             shuffle (bool): Whether to shuffle the data or not. Default is False.
#         """
#         self.feature_folder = feature_folder
#         self.shuffle = shuffle
#         paths = glob.glob(os.path.join(self.feature_folder, '*.h5'))
#         self.paths = [i for i in paths if os.path.basename(i).split(".h5")[0] in basename_list]
#         self.indices = np.arange(len(self.paths))
#         self.transform = transform
#         # Store mean and std for normalization
#         self.mean = torch.tensor(mean, dtype=torch.float32) if mean is not None else None
#         self.std = torch.tensor(std, dtype=torch.float32) if std is not None else None 
#         if self.shuffle:
#             np.random.shuffle(self.indices)

#     def get_feature_path(self, basename):
#         return os.path.join(self.feature_folder,
#                             f'{basename}.h5')

#     def __len__(self):
#         """Returns the total number of samples."""
#         return len(self.indices)


#     def __getitem__(self, index):
#         """
#         Args:
#             index (int): Index of the sample.

#         Returns:
#             dict: {
#                 'features': Tensor of features,
#                 'label': Tensor of label,
#                 'patch_indices': Tensor of patch indices,
#                 'coordinates': Tensor of coordinates,
#                 'spixels_indices': Tensor of superpixel indices,
#                 'file_basename': String file name
#             }
#         """
#         # Get the index of the file
#         file_idx = self.indices[index]
#         file_path = self.paths[file_idx]
#         file_basename = os.path.basename(file_path).split(".")[0]  # Extract filename without extension

#         # Load the HDF5 file
#         with h5py.File(file_path, "r") as f:
#             features = f['features'][:]
#             patch_indices = f['patch_indices'][:]  # Patch indices
#             label = f['label'][()]  # Extract scalar value
#             coordinates = f['coordinates'][:]
#             spixels_indices = f['spixel_idx'][:]

#         # Convert to tensors
#         sample = {
#             "features": torch.tensor(features, dtype=torch.float32),
#             "label": torch.tensor(label, dtype=torch.float32).view(1),  # Ensure label is scalar
#             "patch_indices": torch.tensor(patch_indices, dtype=torch.int64),
#             "coordinates": torch.tensor(coordinates, dtype=torch.float32),
#             "spixels_indices": torch.tensor(spixels_indices, dtype=torch.int64),
#             "file_basename": file_basename  # Keep as string
#         }
#         # Normalize features
#         if self.mean is not None and self.std is not None:
#             sample["features"] = (sample["features"] - self.mean) / (self.std + 1e-8)  # Avoid division by zero 
#         # Apply transformation (if any)
#         if self.transform:
#             sample["features"] = self.transform(sample["features"])

#         return sample
# # scp -r camelyon_csv_splits/* mvu9@maui.rcdc.uh.edu:/project/hnguyen2/mvu9/camelyon16/camelyon_csv_splits/*
