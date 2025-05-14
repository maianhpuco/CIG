import torch
import h5py 
import saliency.core as saliency 
import numpy as np 
from tqdm import tqdm 
import random 

def PreprocessInputs(inputs):
    """ Convert inputsa to a PyTorch tensor and enable gradient tracking """
    inputs = torch.tensor(inputs, dtype=torch.float32).clone().detach()
    return inputs.requires_grad_(True)
 
def call_model_function(images, model, call_model_args=None, expected_keys=None):
    """ Compute model logits and gradients """
    images = PreprocessInputs(images)
    model.eval()
    logits = model(images, [images.shape[0]])
    output = logits
    grads = torch.autograd.grad(output, images, grad_outputs=torch.ones_like(output), create_graph=False)
    gradients = grads[0].detach().cpu().numpy()
    return {saliency.base.INPUT_OUTPUT_GRADIENTS: gradients}


def get_mean_std_for_normal_dist(dataset):
    # Initialize accumulators
    feature_sum = None
    feature_sq_sum = None
    total_samples = 0

    for i in tqdm(range(len(dataset)), desc="Computing the Mean and Std"):
        sample = dataset[i]
        features = torch.tensor(sample['features'], dtype=torch.float32)  # Convert to tensor

        if feature_sum is None:
            feature_sum = torch.zeros_like(features.sum(dim=0))
            feature_sq_sum = torch.zeros_like(features.sum(dim=0))

        feature_sum += features.sum(dim=0)
        feature_sq_sum += (features ** 2).sum(dim=0)
        total_samples += features.shape[0]  # Number of patches

    # Compute mean and std
    mean = feature_sum / total_samples
    std = torch.sqrt((feature_sq_sum / total_samples) - (mean ** 2))
    
    return mean, std


def sample_random_features(dataset, num_files=10):
    basenames = dataset.basenames[:]
    random.shuffle(basenames)
    selected_basenames = random.sample(basenames, min(num_files, len(dataset)))
    stacked_features = []
    with tqdm(total=len(selected_basenames), desc="Loading Feature Files") as pbar:
        for basename in selected_basenames:
            sample = dataset[dataset.basenames.index(basename)]  # Get dataset sample

            features = torch.tensor(sample['features'], dtype=torch.float32)  # Convert to tensor
            stacked_features.append(features)

            pbar.update(1)

    # Stack all features together along the first dimension
    stacked_features = torch.cat(stacked_features, dim=0)
    return stacked_features, selected_basenames
