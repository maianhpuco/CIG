import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob
import openslide 
import json 
import cv2 
import time 

import torch
from torch.utils.data import Dataset, DataLoader
import os
import h5py
import openslide

class IG_dataset(Dataset):
    def __init__(self, features_h5_path, slide_path, basenames=[]): 
        if len(basenames) != 0:
            self.basenames = basenames
        else: 
            self.basenames = [i.split(".")[0] for i in os.listdir(slide_path) if f'{i.split(".")[0]}.h5' in os.listdir(features_h5_path)] 
        self.feature_h5_path = features_h5_path
        
    def __len__(self):
        return len(self.basenames)
    
    def __getitem__(self, idx):
        basename = self.basenames[idx]
        file_path = os.path.join( self.feature_h5_path, f'{basename}.h5') 
        result = {} 
        with h5py.File(file_path, "r") as f:
            result['basename']=basename
            result['file_name'] = file_path 
            result['features'] = f['features'][:]
            result['patch_indices'] = f['patch_indices'][:]
            result['label'] = f['label'][()]
            result['coordinates'] = f['coordinates'][:]
            result['spixel_idx'] = f['spixel_idx'][:] 
        return result 
            
