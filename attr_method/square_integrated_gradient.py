import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch

def normalize_by_2norm(x):
    batch_size = x.shape[0]
    norm = np.sqrt(np.sum(np.power(x, 2).reshape(batch_size, -1), axis=1))  # L2 norm

    norm = np.where(norm == 0, 1e-8, norm)  # Avoid division by zero
    normed_x = np.moveaxis(x, 0, -1) / norm
    return np.moveaxis(normed_x, -1, 0) 

class SquareIntegratedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 25)
        eta = kwargs.get("eta", 1 ) 
        
        attribution_values =  np.zeros_like(x_value, dtype=np.float32)
        alphas = np.linspace(0, 1, x_steps)
        
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]  
        x_diff = x_value - x_baseline_batch   
        
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)

            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
                expected_keys=self.expected_keys
            )
            self.format_and_check_call_model_output(call_model_output, x_step_batch_tensor.shape, self.expected_keys)
            
            baseline_num = 1 
            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(baseline_num, x_value.shape[0], x_value.shape[1])
            
            # ------------ Counter Factual Gradient ------------
            x_baseline_torch = torch.tensor(x_baseline_batch.copy(), dtype=torch.float32, requires_grad=False)
            logits_x_r = model(x_baseline_torch, [x_baseline_torch.shape[0]])

            # Compute counterfactual gradients using logits difference
            x_step_batch_torch = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)
            logits_x_step = model(x_step_batch_torch, [x_step_batch_torch.shape[0]])
            logits_difference = torch.norm(logits_x_step - logits_x_r, p=2) ** 2
            logits_difference.backward()
            
            if x_step_batch_torch.grad is None:
                raise RuntimeError("Gradients are not being computed! Ensure tensors require gradients.")

            grad_logits_diff = x_step_batch_torch.grad.numpy()
        
            # ------------ Combine Gradient and X_diff ------------ 
            gradients_avg = gradients_batch.mean(axis=0) 
            counterfactual_gradients = grad_logits_diff.mean(axis=0) 
            
            W_j = np.linalg.norm(gradients_avg) + 1e-8   
            
            attribution_values += (counterfactual_gradients * gradients_avg) * (eta / W_j) 

        return attribution_values / x_steps 