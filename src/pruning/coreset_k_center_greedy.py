import numpy as np
from sklearn.metrics import pairwise_distances
from sklearn.cluster import KMeans

def k_center_greedy(args):
    """
    Select core-set points using the K-Center Greedy algorithm.

    Args:
        features (numpy.ndarray): Feature matrix of shape (n_samples, n_features).
        n_samples (int): Number of points to select.

    Returns:
        selected_indices (list): Indices of selected points.
    """
    features = args.features
    n_samples = args.samples 
    
    n_points = features.shape[0]

    # Step 1: Calculate pairwise distances between all points in the dataset
    # This creates a matrix where each entry represents the distance between two points.
    distances = pairwise_distances(features, features, metric="euclidean")

    # Step 2: Randomly select the first point to start the selection process
    selected_indices = [np.random.choice(range(n_points))]

    # Step 3: Initialize minimum distances to the first selected point
    min_distances = distances[selected_indices[0]]

    # Step 4: Iteratively select `n_samples - 1` more points
    for _ in range(n_samples - 1):
        # Find the point that is farthest from the currently selected points
        farthest_point = np.argmax(min_distances)
        selected_indices.append(farthest_point)

        # Update the minimum distances to include the new point
        min_distances = np.minimum(min_distances, distances[farthest_point])

    # Step 5: Return the indices of the selected core-set points
    return selected_indices
