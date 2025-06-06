Contrastive Integrated Gradients (CIG)

This repository contains the implementation of Contrastive Integrated Gradients (CIG), a gradient-based attribution method designed for weakly supervised classification tasks such as Whole Slide Image (WSI) analysis. CIG estimates feature importance by measuring the change in model predictions as input features are interpolated between the original sample and a counterfactual baseline sampled from the opposite class.

## How to use 
###  Import the Module
```
from attr_method.cig import CIG, ModelWrapper, call_model_function
``` 
###  Prepare Inputs
- x_value: Feature tensor for the slide you want to explain. Shape: [1, N, D]
- baseline_features: Feature tensor sampled from the opposite class. Shape: [N, D] 


### Run CIG 
```
cig = CIG()

attr_map = cig.GetMask(
    x_value=x_value,
    baseline_features=baseline_features,
    model=model,
    call_model_function=call_model_function,
    call_model_args={"target_class_idx": 1},  # Adjust as needed
    x_steps=50,
    device="cuda"
)
 
```

Output: 
```attr_map```: A NumPy array of shape [N, D], representing the attribution scores for each patch-level feature. 