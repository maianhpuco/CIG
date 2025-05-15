import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch

class IntegratedDecisionGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""
    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    @staticmethod
    def getSlopes(x_baseline_batch, x_value, model, x_steps):
        # Generate alpha values (steps,)
        alphas = np.linspace(0, 1, x_steps) 

        logits = torch.zeros(x_steps)
        slopes = torch.zeros(x_steps)

        x_diff = x_value - x_baseline_batch 

        # Compute logits for each interpolation step
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing IG", ncols=100), start=0):
            # Compute interpolated input
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)  
            # Get model output logits
            logit = model(x_step_batch_tensor, [x_step_batch_tensor.shape[0]]).detach().cpu().numpy()
            
            logit = logit.item()
            logits[step_idx] = logit  

        # Compute x_diff (alpha step difference) for slopes
        x_diff_value = float(alphas[1] - alphas[0])  # Constant for uniform steps 

        # Compute slopes using finite differences
        slopes[1:] = (logits[1:] - logits[:-1]) / x_diff_value 

        # Ensure the first slope is zero
        slopes[0] = 0

        return slopes, x_diff_value, logits 
    
    @staticmethod
    def getAlphaParameters(slopes, steps, step_size):
        # normalize slopes 0 to 1 to eliminate negatives and preserve magnitude
        slopes_0_1_norm = (slopes - torch.min(slopes)) / (torch.max(slopes) - torch.min(slopes))
        # reset the first slope to zero after normalization because it is impossible to be nonzero
        slopes_0_1_norm[0] = 0
        # normalize the slope values so that they sum to 1.0 and preserve magnitude
        slopes_sum_1_norm = slopes_0_1_norm / torch.sum(slopes_0_1_norm)

        # obtain the samples at each alpha step as a float based on the slope (steps/alpha)
        sample_placements_float = torch.mul(slopes_sum_1_norm, steps)
        # truncate the result to int values to clean up decimals, this leaves unused steps (samples)
        sample_placements_int = sample_placements_float.type(torch.int)
        # find how many unused steps are left
        remaining_to_fill = steps - torch.sum(sample_placements_int)

        # find the values which were not truncated to 0 (float values >= 1) 
        # by the int casting and make them -1 in the float array
        non_zeros = torch.where(sample_placements_int != 0)[0]
        sample_placements_float[non_zeros] = -1

        # Find the indicies of the remaining spots to fill from the float array (the zero values) sorted high to low
        remaining_hi_lo = torch.flip(torch.sort(sample_placements_float)[1], dims = [0])
        # Fill all of these spots in the int array with 1, this gives the final distribution of steps
        sample_placements_int[remaining_hi_lo[0 : remaining_to_fill]] = 1

        # holds new alpha values to be created
        alphas = torch.zeros(steps)    
        # an array that tracks indivdual steps between alpha values
        # this is important to counteract the non-uniform alpha spacing of this method
        alpha_substep_size = torch.zeros(steps)

        # the index at which a range of samples begins, it is a function of num_samples in loop
        alpha_start_index = 0
        # the value at which a range of samples starts, it is a function of step_size
        alpha_start_value = 0

        # generate the new alpha values
        for num_samples in sample_placements_int:        
            if num_samples == 0:
                continue

            # Linearly divide the samples into the required alpha range
            alphas[alpha_start_index: (alpha_start_index + num_samples)] = torch.linspace(alpha_start_value, alpha_start_value + step_size, num_samples + 1)[0 : num_samples]

            # track the step size of the alpha divisions
            alpha_substep_size[alpha_start_index: (alpha_start_index + num_samples)] = (step_size / num_samples)

            alpha_start_index += num_samples
            alpha_start_value += step_size

        return alphas, alpha_substep_size  
    
    def GetMask(self, **kwargs):
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 25)
                
        sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
        x_baseline_batch = baseline_features[sampled_indices]       
        x_baseline_batch_flat = x_baseline_batch.reshape(-1, x_baseline_batch.shape[-1]) 
        x_value_flat = x_value.reshape(-1, x_value.shape[-1])  
        
        slopes, x_diff_value, logits = self.getSlopes(x_baseline_batch_flat, x_value_flat, model, x_steps)  
        new_alphas, alpha_substep_size = self.getAlphaParameters(slopes, x_steps, 1/x_steps)
        alphas_np = new_alphas.numpy()
        alpha_substep_size_np = alpha_substep_size.numpy()
        
        # ----- Compute integrated Gradient 
        _integrated_gradient =  np.zeros_like(x_value, dtype=np.float32)  
        prev_logit = None  # Initialize previous logits for slope computation
        slopes = np.zeros(x_steps) 
        x_diff = x_value - x_baseline_batch 
        
        for step_idx, alpha in enumerate(tqdm(alphas_np, desc="Computing IG²", ncols=100), start=1):
            x_step_batch = x_baseline_batch + alpha * x_diff
            x_step_batch_tensor = torch.tensor(x_step_batch, dtype=torch.float32, requires_grad=True)

            call_model_output = call_model_function(
                x_step_batch_tensor,
                model,
                call_model_args=call_model_args,
            )
            
            logit = model(x_step_batch_tensor.squeeze(0), [x_step_batch_tensor.squeeze(0).shape[0]])

            gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(1, x_value.shape[0], x_value.shape[1])
            gradients_avg = np.mean(gradients_batch, axis=0)
            idx = step_idx - 1 
            if prev_logit is not None:  # Skip first step since there's no previous logit
                slopes[idx] = (logit - prev_logit) / (alpha - alphas_np[idx - 1])  # alpha difference 

            # compute slope 
            prev_logit = logit  

            weighted_grad = gradients_avg * slopes[idx]
            weighted_grad = weighted_grad * alpha_substep_size_np[idx]
            
            _integrated_gradient += weighted_grad
            
        attribution_values = _integrated_gradient * x_diff.reshape(-1, x_value.shape[1])
        
        return attribution_values