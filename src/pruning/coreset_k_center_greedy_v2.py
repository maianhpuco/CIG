import torch
import numpy as np
from sklearn.metrics import pairwise_distances

# def k_center_greedy(features, patch_indices, pruning_rate=0.1):
#     """
#     Select core-set points using the K-Center Greedy algorithm based on a fraction.
    
#     Args:
#         features (torch.Tensor): Feature matrix of shape (n_samples, n_features).
#         patch_indices (torch.Tensor): Patch indices corresponding to the features.
#         pruning_rate (float): Fraction of points to select (0.0 to 1.0).
    
#     Returns:
#         selected_features (torch.Tensor): Features corresponding to the selected points.
#         selected_patch_indices (torch.Tensor): Patch indices corresponding to the selected points.
#         selected_indices (torch.Tensor): Indices of selected points.
#     """
#     n_points = features.shape[0]
#     n_samples_to_select = int(n_points * (1 - pruning_rate))
#     distances = pairwise_distances(features.numpy(), features.numpy(), metric="euclidean")
#     selected_indices = [np.random.choice(range(n_points))]
#     min_distances = distances[selected_indices[0]]

#     for _ in range(n_samples_to_select - 1):
#         farthest_point = np.argmax(min_distances)
#         selected_indices.append(farthest_point)
#         min_distances = np.minimum(min_distances, distances[farthest_point])

#     selected_indices = torch.tensor(selected_indices, dtype=torch.long)
#     selected_features = features[selected_indices]
#     selected_patch_indices = patch_indices[selected_indices]

#     return selected_features, selected_patch_indices 

import torch
import numpy as np
from sklearn.metrics import pairwise_distances

def k_center_greedy(features, patch_indices, pruning_rate=0.1):
    """
    Select core-set points using the K-Center Greedy algorithm based on a fraction.
    
    Args:
        features (torch.Tensor): Feature matrix of shape (n_samples, n_features).
        patch_indices (torch.Tensor): Patch indices corresponding to the features.
        pruning_rate (float): Fraction of points to select (0.0 to 1.0).
    
    Returns:
        selected_features (torch.Tensor): Features corresponding to the selected points.
        selected_patch_indices (torch.Tensor): Patch indices corresponding to the selected points.
        selected_indices (torch.Tensor): Indices of selected points.
    """
    n_points = features.shape[0]
    
    # Calculate the number of points to select based on the pruning_rate
    n_samples_to_select = int(n_points * (1 - pruning_rate))

    # Step 1: Calculate pairwise distances between all points in the dataset
    distances = pairwise_distances(features, features, metric="euclidean")

    # Step 2: Randomly select the first point to start the selection process
    selected_indices = [np.random.choice(range(n_points))]

    # Step 3: Initialize minimum distances to the first selected point
    min_distances = distances[selected_indices[0]]

    # Step 4: Iteratively select `n_samples_to_select - 1` more points
    for _ in range(n_samples_to_select - 1):
        # Find the point that is farthest from the currently selected points
        farthest_point = np.argmax(min_distances)
        selected_indices.append(farthest_point)

        # Update the minimum distances to include the new point
        min_distances = np.minimum(min_distances, distances[farthest_point])

    # Convert selected indices to PyTorch tensor
    # selected_indices = torch.tensor(selected_indices, dtype=torch.long)

    # Step 5: Return the selected features and patch indices
    selected_features = features[selected_indices]
    selected_patch_indices = patch_indices[selected_indices]

    return selected_features, selected_patch_indices
