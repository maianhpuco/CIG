import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob 

class FeaturesDataset(Dataset):
    def __init__(
        self, 
        feature_folder, 
        basename_list=None, 
        feature_file_end ='h5',  
        shuffle=True,
        transform=None 
        ):
        """
        Args:
            label_file (str): Path to the CSV file containing labels for each sample.
            shuffle (bool): Whether to shuffle the data or not. Default is False.
        """
        self.feature_folder = feature_folder
        self.shuffle = shuffle
        paths = glob.glob(os.path.join(self.feature_folder, '*.h5'))
        self.paths = [i for i in paths if os.path.basename(i).split(".h5")[0] in basename_list]
        self.indices = np.arange(len(self.paths))
        self.transform = transform 
         
        if self.shuffle:
            np.random.shuffle(self.indices)
        
    def get_feature_path(self, basename):
        return os.path.join(self.feature_folder, 
                            f'{basename}.h5')
    
    def __len__(self):
        """Returns the total number of samples."""
        return len(self.indices)

    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample.
        
        Returns:
            tuple: (feature, adjacency matrix, label)
        """
        # Get the index of the file
        file_idx = self.indices[index]
        file_path = self.paths[file_idx]
        file_basename = os.path.basename(file_path).split(".")
        
        # Load the HDF5 file
        with h5py.File(file_path,  "r") as f:
            features = f['features'][:]
            patch_indices = f['patch_indices'][:] # how to get these indice 
            label = f['label'][:]
            coordinates = f['coordinates'][:] 
            spixels_indices = f['spixel_idx'][:] 

        label_tensor = torch.tensor([label], dtype=torch.float32).view(1, 1)
        features_tensor = torch.tensor(features, dtype=torch.float32)
        patch_indices = torch.tensor(patch_indices, dtype=torch.int64)
        coordinates = torch.tensor(coordinates, dtype=torch.float32) 
        spixels_indices = torch.tensor(spixels_indices, dtype=torch.int64) 
        # features_tensor = self.transform(features_tensor)
        if self.transform:
            features_tensor = self.transform(features_tensor)
         
        return features_tensor, label_tensor, patch_indices, coordinates, spixels_indices, file_basename 
