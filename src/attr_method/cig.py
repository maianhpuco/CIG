import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import torch.nn.functional as F


class ModelWrapper:
    """Wraps a model to provide a unified forward interface for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        # If input is batched as [1, N, D], remove the batch dimension
        if input.dim() == 3:
            input = input.squeeze(0)

        if self.model_type == 'clam':
            # CLAM model returns a tuple; extract logits
            output = self.model(input, [input.shape[0]])
            logits = output[0] if isinstance(output, tuple) else output
        else:
            # For other models, assume direct output
            logits = self.model(input)
        return logits 
    

# Helper to call model and compute gradients if requested
def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3
        inputs = inputs.to(device)
        
        if not inputs.requires_grad:
            inputs.requires_grad_(True)  # Enable gradient computation if not already set

        if was_batched:
            inputs = inputs.squeeze(0)

        model.eval()
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(inputs)

        if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
            # If gradient is requested, compute gradient w.r.t. target class
            class_idx = call_model_args.get("target_class_idx", 0)
            target = logits[:, class_idx]
            grads = torch.autograd.grad(
                outputs=target,
                inputs=inputs,
                grad_outputs=torch.ones_like(target),
                retain_graph=False,
                create_graph=False
            )[0]
            grads_np = grads.detach().cpu().numpy()
            if was_batched or grads_np.ndim == 2:
                grads_np = np.expand_dims(grads_np, axis=0)
            return {INPUT_OUTPUT_GRADIENTS: grads_np}

        return logits 


class CIG(CoreSaliency):
    """Contrastive Integrated Gradients for efficient attribution using a counterfactual baseline."""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")  # Input features: tensor or numpy array [1, N, D]
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features", None)  # Baseline: tensor or numpy array [N, D]
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert to torch tensors if needed
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        # Ensure baseline shape matches number of patches/features
        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        # Add batch dimension to baseline
        baseline_features = baseline_features.unsqueeze(0)  # Shape: [1, N, D]

        # Initialize attributions
        attribution_values = torch.zeros_like(x_value, device=device)

        # Compute differences between input and baseline
        x_diff = x_value - baseline_features

        # Create interpolation steps (skip alpha=0 to avoid duplicate baseline)
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]

        # Attribution loop
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = baseline_features + alpha * x_diff  # Interpolated input
            x_step_batch.requires_grad_(True)
            x_step_batch.retain_grad()

            # Get logits for baseline and current step
            logits_r = call_model_function(baseline_features, model, call_model_args)
            if isinstance(logits_r, tuple):
                logits_r = logits_r[0]

            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            # Loss: squared L2 distance between logits
            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()

            gradients = x_step_batch.grad 
            if gradients is None:
                print(f">  No gradients at alpha {alpha:.2f}, skipping")
                continue

            # Average over batch dimension if needed
            counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients

            # Accumulate gradient
            attribution_values += counterfactual_gradients

        # Final scaling by average difference (elementwise multiplication)
        x_diff_mean = x_diff.mean(dim=0)
        attribution_values *= x_diff_mean

        # Normalize and return result as numpy
        return attribution_values.detach().cpu().numpy() / x_steps 
