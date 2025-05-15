import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity_matrix_importance(features_np):
    """
    Compute importance scores using the similarity matrix.
    """
    similarity_matrix = 1 - cosine_similarity(features_np)
    importance_scores = np.mean(similarity_matrix, axis=1)
    return importance_scores

def importance_based_feature_selection(features, patch_indices, pruning_rate=0.1):
    """
    Select features based on their importance scores computed from the similarity matrix.

    Args:
        features (np.ndarray): Feature matrix of shape (n_samples, n_features).
        patch_indices (np.ndarray): Indices corresponding to the features.
        importance_scores (np.ndarray): Importance scores computed from similarity matrix.
        pruning_rate (float): Fraction of features to select (0.0 to 1.0).

    Returns:
        np.ndarray: Selected features based on importance.
        np.ndarray: Corresponding patch indices for selected features.
    """
    importance_scores = compute_similarity_matrix_importance(
        features 
    )
    # Sort indices based on importance scores (descending order)
    sorted_indices = np.argsort(importance_scores)[::-1]

    # Determine how many features to select based on the pruning rate
    total_samples = len(features)
    n_to_select = int(total_samples * (1 - pruning_rate))

    # Select the top `n_to_select` indices based on importance scores
    selected_indices = sorted_indices[:n_to_select]

    # Slice the features and patch_indices using the selected indices
    _features = features[selected_indices]
    _patch_indices = patch_indices[selected_indices]
    
    return _features, _patch_indices

