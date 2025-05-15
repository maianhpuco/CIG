import numpy as np
from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch

class ExpectedGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)
        baseline_features = kwargs.get("baseline_features", None)
        x_steps = kwargs.get("x_steps", 25) 
        
        attribution_values =  np.zeros_like(x_value, dtype=np.float32)

        alphas = np.random.uniform(low=0.0, high=1.0, size=x_steps) 
        for step_idx, alpha in enumerate(tqdm(alphas, desc="Computing:", ncols=100), start=1):
            sampled_indices = np.random.choice(baseline_features.shape[0], (1, x_value.shape[0]), replace=True)
            x_baseline_batch = baseline_features[sampled_indices]     
            x_diff = x_value - x_baseline_batch   
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
            
            gradients_avg = gradients_batch.reshape(-1, x_value.shape[-1])
            
            x_diff = x_diff.reshape(-1, x_value.shape[-1])          
            
            attribution_values += (gradients_avg * x_diff) 
            
        return attribution_values / x_steps