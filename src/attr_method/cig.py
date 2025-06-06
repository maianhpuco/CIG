import os
import numpy as np
import saliency.core as saliency
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch
import torch.nn.functional as F


class ModelWrapper:
    """Wraps a model to standardize forward calls for different model types."""
    def __init__(self, model, model_type: str = 'clam'):
        self.model = model
        self.model_type = model_type.lower()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            input = input.squeeze(0)
        if self.model_type == 'clam':
            output = self.model(input, [input.shape[0]])
            logits = output[0] if isinstance(output, tuple) else output # for example CLAM
        else:
            logits = self.model(input)
        return logits 
    
    
# from attr_method._common import PreprocessInputs, call_model_function
def call_model_function(inputs, model, call_model_args=None, expected_keys=None):
        device = next(model.parameters()).device
        was_batched = inputs.dim() == 3
        inputs = inputs.to(device)
        
        if not inputs.requires_grad:
            inputs.requires_grad_(True) 
            
        # inputs = inputs.to(device).clone().detach().requires_grad_(True)
        if was_batched:
            inputs = inputs.squeeze(0)

        model.eval()
        model_wrapper = ModelWrapper(model, model_type='clam')
        logits = model_wrapper.forward(inputs)

        if expected_keys and INPUT_OUTPUT_GRADIENTS in expected_keys:
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
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")  # numpy array [1, N, D] or tensor
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", {})
        baseline_features = kwargs.get("baseline_features", None)  # numpy array or tensor [N, D]
        x_steps = kwargs.get("x_steps", 25) 
        device = kwargs.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        # Convert inputs to torch tensors if necessary
        if isinstance(x_value, np.ndarray):
            x_value = torch.tensor(x_value, dtype=torch.float32)
        if isinstance(baseline_features, np.ndarray):
            baseline_features = torch.tensor(baseline_features, dtype=torch.float32)

        x_value = x_value.to(device)
        baseline_features = baseline_features.to(device)

        if baseline_features.shape[0] != x_value.shape[1] or baseline_features.shape[1] != x_value.shape[2]:
            raise ValueError(f"Baseline shape {baseline_features.shape} does not match x_value patch dimension {x_value.shape[1:]}")

        baseline_features = baseline_features.unsqueeze(0)  # Shape: [1, N, D]
        attribution_values = torch.zeros_like(x_value, device=device)

        x_diff = x_value - baseline_features
        alphas = torch.linspace(0, 1, x_steps + 1, device=device)[1:]  # Skip alpha=0

        # Precompute reference logits

        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = baseline_features + alpha * x_diff  # [1, N, D]
            x_step_batch.requires_grad_(True)  # Ensure it's a leaf with grad
            x_step_batch.retain_grad()

            # Forward pass - no torch.no_grad() here
            logits_r = call_model_function(baseline_features, model, call_model_args)
            if isinstance(logits_r, tuple):
                logits_r = logits_r[0]
            # logits_r = logits_r.detach()  # detach so it's not part of graph

            logits_step = call_model_function(x_step_batch, model, call_model_args)
            if isinstance(logits_step, tuple):
                logits_step = logits_step[0]

            loss = torch.norm(logits_step - logits_r, p=2) ** 2
            loss.backward()
            gradients = x_step_batch.grad 
            if gradients is None:
                print(f">>>>>>>>>>>>>>>>>> No gradients at alpha {alpha:.2f}, skipping")
                continue

            counterfactual_gradients = gradients.mean(dim=0) if gradients.dim() > 2 else gradients
            attribution_values += counterfactual_gradients

        x_diff_mean = x_diff.mean(dim=0)
        attribution_values *= x_diff_mean
        return attribution_values.detach().cpu().numpy() / x_steps