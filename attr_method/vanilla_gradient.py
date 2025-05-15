from tqdm import tqdm
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
import torch

class VanillaGradients(CoreSaliency):
    """Efficient Integrated Gradients with Counterfactual Attribution"""

    expected_keys = [INPUT_OUTPUT_GRADIENTS]

    def GetMask(self, **kwargs): 
        x_value = kwargs.get("x_value")
        call_model_function = kwargs.get("call_model_function")
        model = kwargs.get("model") 
        call_model_args = kwargs.get("call_model_args", None)

        x_value_tensor = torch.tensor(x_value, dtype=torch.float32, requires_grad=True) 
        
        call_model_output = call_model_function(
            x_value_tensor,
            model,
            call_model_args=call_model_args,
            expected_keys=self.expected_keys
        ) 
        # print("all_model_output.shape", call_model_output.shape) 
        self.format_and_check_call_model_output(call_model_output, x_value_tensor.shape, self.expected_keys) 
    
        gradients_batch = call_model_output[INPUT_OUTPUT_GRADIENTS].reshape(1, x_value.shape[0], x_value.shape[1]) 
        gradients = gradients_batch.reshape(-1, x_value.shape[-1]) 
         
        return gradients
        
