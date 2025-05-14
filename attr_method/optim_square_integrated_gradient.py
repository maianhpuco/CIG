import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import matplotlib.pyplot as plt
# from attr_method._common import PreprocessInputs, call_model_function 


def normalize_by_2norm(x):
    """Normalize gradients using L2 norm."""
    batch_size = x.shape[0]
    norm = np.sqrt(np.sum(np.power(x, 2).reshape(batch_size, -1), axis=1))  # L2 norm
    norm = np.where(norm == 0, 1e-8, norm)  # Avoid division by zero
    normed_x = np.moveaxis(x, 0, -1) / norm
    return np.moveaxis(normed_x, -1, 0) 


class OptimSquareIntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        """Computes the integrated gradient attributions using GradPath (IG²)."""
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model")
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 5)
        eta = kwargs.get("eta", 1 ) 
        memmap_path = kwargs.get("memmap_path")
        

        # Sample reference points (baseline)
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices] 

        # 🔥 **Compute GradPath (returns path and counterfactual gradients)**
        path, counterfactual_gradients_memmap = self.Get_GradPath(x_value, x_baseline_batch, model, x_steps, memmap_path)

        # Ensure first step in path is close to original input
        np.testing.assert_allclose(x_value, path[0], rtol=0.01)

        # 🔥 **Integrate Gradients on GradPath (IG²)**
        print('Integrating gradients on GradPath...')
        attr = np.zeros_like(x_value, dtype=np.float32)
        x_old = x_value

        for i, x_step in enumerate(path[1:]):
            call_model_output = call_model_function(
                x_old,
                model, 
                call_model_args=call_model_args,
                expected_keys=[INPUT_OUTPUT_GRADIENTS]
                )
            self.format_and_check_call_model_output(call_model_output, x_old.shape, self.expected_keys)
            
            baseline_num = 1 
            feature_gradient = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(baseline_num, x_value.shape[0], x_value.shape[1]) 
            feature_gradient = feature_gradient.mean(axis=0) 
            # print("feature_gradient", feature_gradient.shape)

            #  Feature Gradient: dF/dX
            # feature_gradient = call_model_output[INPUT_OUTPUT_GRADIENTS]

            #  Retrieve Precomputed Counterfactual Gradient
            counterfactual_gradient = counterfactual_gradients_memmap[i]
            # print("counterfactual_gradient", counterfactual_gradient.shape) 
            # print("(x_old - x_step).shape", (x_old - x_step).shape)
            #  Apply IG² Equation
            W_j = np.linalg.norm(feature_gradient) + 1e-8  # Avoid division by zero
            attr += (x_old - x_step) * feature_gradient * counterfactual_gradient * (eta / W_j)
            x_old = x_step  # Move to next step
        return attr
     
    @staticmethod
    def Get_GradPath(x_value, baselines, model, x_steps, memmap_path):
        """Computes the iterative GradPath using gradient-based perturbations."""

        # Initialize memmap for storing GradPath and Counterfactual Gradients
        path_filename = f"{memmap_path}/grad_path_memmap.npy"
        counterfactual_grad_filename = f"{memmap_path}/counterfactual_gradients_memmap.npy"
        path_shape = (x_steps, *x_value.shape)  
        grad_shape = (x_steps-1, *x_value.shape)  # One less step for gradients

        path_memmap = np.memmap(path_filename, dtype=np.float32, mode='w+', shape=path_shape)
        counterfactual_gradients_memmap = np.memmap(counterfactual_grad_filename, dtype=np.float32, mode='w+', shape=grad_shape)

        # Compute logits for baseline (counterfactual reference)
        x_baseline_torch = torch.tensor(baselines.copy(), dtype=torch.float32, requires_grad=False)
        logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])

        # Initialize perturbation delta
        delta = np.zeros_like(x_value)
        path_memmap[0] = x_value  

        progress_bar = tqdm(range(1, x_steps), desc="Searching GradPath", ncols=100)
        step_size = 1.0  # Initial step size
        prev_loss = float('inf')

        for i in progress_bar:
            # Perturb the input
            x_step_batch = x_value + delta
            x_step_batch_torch = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)

            # Compute logits for perturbed input
            logits_x_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])

            # Compute counterfactual gradient: Norm of logits difference
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2

            # Compute Counterfactual Gradients Once and Store
            grad_logits_diff = torch.autograd.grad(logits_difference, x_step_batch_torch, retain_graph=True)[0]

            if grad_logits_diff is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            # Convert gradients to numpy and normalize
            grad_logits_diff = grad_logits_diff.detach().cpu().numpy()  
            grad_logits_diff = normalize_by_2norm(grad_logits_diff)

            # Store counterfactual gradients in memmap
            counterfactual_gradients_memmap[i-1] = grad_logits_diff  # i-1 since gradient is computed after 1st step

            # Adjust step size dynamically
            if logits_difference.item() < prev_loss:
                step_size *= 1.1  # Increase step size if loss is decreasing
            else:
                step_size *= 0.9  # Reduce step size smoothly

            prev_loss = logits_difference.item()

            # Update delta using normalized gradients (No Clipping)
            with torch.no_grad():
                delta = delta + grad_logits_diff * step_size
                x_adv = x_value + delta  # Update perturbed input

            # Save new perturbed input in memmap (disk instead of RAM)
            path_memmap[i] = x_adv
            path_memmap.flush()  
            counterfactual_gradients_memmap.flush()  

            # Update tqdm progress bar dynamically
            progress_bar.set_postfix({"Loss": logits_difference.item(), "Step Size": step_size})

            # Free memory of tensors no longer needed
            del x_step_batch_torch, logits_x_step, logits_difference
            torch.cuda.empty_cache()  

        return path_memmap, counterfactual_gradients_memmap  