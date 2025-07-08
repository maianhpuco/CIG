# Contrastive Integrated Gradients (CIG)

This repository contains the implementation of Contrastive Integrated Gradients (CIG), a gradient-based attribution method designed for weakly supervised classification tasks such as Whole Slide Image (WSI) analysis. CIG estimates feature importance by measuring the change in model predictions as input features are interpolated between the original sample and a counterfactual baseline sampled from the opposite class.

##  Installation 
Clone this repository and install dependencies: 
```
git clone https://github.com/yourusername/contrastive-integrated-gradients.git
cd contrastive-integrated-gradients
pip install -r requirements.txt ```

## How to Use 

Import the Module 
```
from attr_method.cig import CIG, ModelWrapper, call_model_function 
``` 
Prepare Inputs
`x_value`: Tensor of features for the WSI (slide) you want to explain.
Shape: [1, N, D]
`baseline_features`: Counterfactual features sampled from a different class (e.g., non-tumor if the input is tumor).
Shape: [N, D]
`model`: The trained model used for prediction (e.g., CLAM model).
`call_model_args`: A dictionary that specifies which class index to explain, e.g., { "target_class_idx": 1 }. 

## Run CIG 
```
cig = CIG()

attr_map = cig.GetMask(
    x_value=x_value,
    baseline_features=baseline_features,
    model=model,
    call_model_function=call_model_function,
    call_model_args={"target_class_idx": 1},  # Replace 1 with your class of interest
    x_steps=50,
    device="cuda"  # or "cpu"
)
``` 

Output: ```attr_map``` 
Each row corresponds to a patch, and each column corresponds to a feature dimension. The values indicate how important each patch-level feature is to the model's prediction, based on the contrastive gradient path.



