import numpy as np
import torch 

def random_feature_selection(features, patch_indices, pruning_rate=0.1):
    total_sample = features.shape[0]
    n_to_select = int(total_sample * (1 - pruning_rate))
    
    # Use np.random.choice for NumPy arrays
    selected_indices = np.random.choice(range(total_sample), size=n_to_select, replace=False)

    # Slice the features and patch_indices using the selected indices
    _features = features[selected_indices]
    _patch_indices = patch_indices[selected_indices]
    
    return _features, _patch_indices