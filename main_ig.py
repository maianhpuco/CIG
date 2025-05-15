import os
import sys 
import torch
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import timm 
import shutil 
# import yaml 
import random 
import numpy as np
from src.bag_classifier.mil_classifier import MILClassifier 
import torch
import torch.nn as nn 
import torch.optim as optim
import saliency.core as saliency
from saliency.core.base import CoreSaliency
from saliency.core.base import INPUT_OUTPUT_GRADIENTS
from torch.utils.data import DataLoader 
from src.bag_classifier.mil_classifier import MILClassifier # in the repo
from data.feature_dataset import FeaturesDataset  # in the repo
from utils.utils import load_config
from utils.train_classifier.train_mlclassifier import (
    save_checkpoint, 
    load_checkpoint
)
import h5py 
import pickle 

from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation
) 
from data.ig_dataset import IG_dataset 
from attr_method._common import (
    sample_random_features, 
    get_mean_std_for_normal_dist, 
    PreprocessInputs, 
    call_model_function
)

def load_model(checkpoint_path):
    input_dim = 768  # Adjust according to dataset
    mil_model = MILClassifier(input_dim=input_dim, pooling='attention')
    optimizer = optim.AdamW(mil_model.parameters(), lr=0.0005)
    model, _, _, _ = load_checkpoint(mil_model, optimizer, checkpoint_path)  
    return model 

def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    #---------------------------------------------------- 
    if args.ig_name=='integrated_gradient':
        from attr_method.integrated_gradient import IntegratedGradients as AttrMethod 
       
    elif args.ig_name=='vanilla_gradient':
        from attr_method.vanilla_gradient import VanillaGradients as AttrMethod 
    
    elif args.ig_name=='contrastive_gradient':
        from attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod 

    elif args.ig_name=='expected_gradient':
       from attr_method.expected_gradient import ExpectedGradients as AttrMethod   
    
    elif args.ig_name=='integrated_decision_gradient':
       from attr_method.integrated_decision_gradient import IntegratedDecisionGradients as AttrMethod     
  
    elif args.ig_name=='optim_square_integrated_gradient':
       from attr_method.optim_square_integrated_gradient import OptimSquareIntegratedGradients as AttrMethod
    
    elif args.ig_name=='square_integrated_gradient':
       from attr_method.square_integrated_gradient import SquareIntegratedGradients as AttrMethod     
    else:
        print(f"Unsupported for {args.ig_name}")
    # LIME
    # KernelSHAP
    # DeepSHAP 
    
    print(f"Running for {args.ig_name} Attribution method") 
    
    
    #----------------------------------------------------    
    attribution_method = AttrMethod()   
    
    score_save_path = os.path.join(args.attribution_scores_folder, f'{args.ig_name}') 
    # print("score_save_path", score_save_path)
    # if os.path.exists(score_save_path):
    #     shutil.rmtree(score_save_path)  # Delete the existing directory
    # os.makedirs(score_save_path)    
    

    checkpoint_path = os.path.join(args.checkpoints_dir, f'{CHECK_POINT_FILE}')
    mil_model = load_model(checkpoint_path)
    
    if args.dry_run==1:
        dataset = IG_dataset(
            args.features_h5_path,
            args.slide_path,
            basenames=['tumor_026', 'tumor_031', 'tumor_032','tumor_036']
        )   
        
    else:
        basenames = [] 
        for basename in os.listdir(args.slide_path):
            basename = basename.split(".")[0]
            if basename.startswith('normal_'): 
            # if basename.startswith(('tumor_', 'test_')):  # Check if it starts with either prefix
                basenames.append(basename)
        
        dataset = IG_dataset(
            args.features_h5_path,
            args.slide_path,
            basenames=basenames
            )
        
    if args.do_normalizing: 
        with h5py.File(args.feature_mean_std_path, "r") as f:
            mean = f["mean"][:]
            std = f["std"][:]
            
    print(">>>>>>>>>----- Total number of sample in dataset:", len(dataset)) 
    
    for idx, data in enumerate(dataset):
        total_file = len(dataset)
        print(f"Processing the file numner {idx+1}/{total_file}")
        basename = data['basename']
        features = data['features']  # Shape: (batch_size, num_patches, feature_dim)
        label = data['label']
        start = time.time() 
            
    
        if args.do_normalizing:   
            print("----- normalizing")
            features = (features - mean) / (std + 1e-8)  
        
        # randomly sampling #file to create the baseline 
        stacked_features_baseline, selected_basenames =  sample_random_features(
            dataset, num_files=20) 
        stacked_features_baseline = stacked_features_baseline.numpy() 
        
        kwargs = {
            "x_value": features,  
            "call_model_function": call_model_function,  
            "model": mil_model,  
            "baseline_features": stacked_features_baseline,  # Optional
            # "memmap_path": args.memmap_path, 
            "x_steps": 50,  
        }  
 
        attribution_values = attribution_method.GetMask(**kwargs) 
        scores = attribution_values.mean(1)
        _save_path = os.path.join(score_save_path, f'{basename}.npy')
        np.save(_save_path, scores)
        print(f"Done save result numpy file at shape {scores.shape} at {_save_path}")
    
         
if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry_run', type=int, default=0)
    parser.add_argument('--config_file', default='ma_exp002')
    parser.add_argument('--ig_name', 
                    default='integrated_gradients', 
                    choices=[
                        'integrated_gradient', 
                        'expected_gradient', 
                        'integrated_decision_gradient', 
                        'contrastive_gradient', 
                        'vanilla_gradient', 
                        'square_integrated_gradient', 
                        'optim_square_integrated_gradient'
                        ],
                    help='Choose the attribution method to use.') 
    parser.add_argument('--attribution_scores_folder', type=str)
    parser.add_argument('--checkpoints_dir', type=str)
    parser.add_argument('--slide_path', type=str)
    parser.add_argument('--features_h5_path', type=str)
    parser.add_argument('--do_normalizing', type=bool)
    
    args = parser.parse_args()
    
    if os.path.exists(f'./testbest_config/{args.config_file}.yaml'):
        config = load_config(f'./testbest_config/{args.config_file}.yaml')
        args.use_features = config.get('use_features', True)
        
        args.slide_path = config.get('SLIDE_PATH')
        # args.json_path = config.get('JSON_PATH')
        # args.spixel_path = config.get('SPIXEL_PATH')
        # args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
        args.features_h5_path = config.get("FEATURES_H5_PATH") # save all the features
        args.checkpoints_dir = config.get("CHECKPOINT_PATH")
        if args.dry_run==1:
            args.attribution_scores_folder = config.get("SCORE_FOLDER_DRYRUN") 
            args.plot_path = config.get("PLOT_PATH_DRYRUN")    
            print("----")
            
            print("args.attribution_scores_folder", args.attribution_scores_folder)
            print("args.plot_path", args.plot_path)
            
        else: 
            args.attribution_scores_folder = config.get("SCORE_FOLDER")    
            args.plot_path = config.get("PLOT_PATH") 
        print("Attribution folder path", args.attribution_scores_folder)
        
        os.makedirs(args.features_h5_path, exist_ok=True)  
        os.makedirs(args.attribution_scores_folder, exist_ok=True) 
        args.batch_size = config.get('batch_size')
        args.feature_extraction_model = config.get('feature_extraction_model')
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
        args.feature_mean_std_path=config.get("FEATURE_MEAN_STD_PATH")
        # args.ig_name = "integrated_gradients"
        args.do_normalizing = True
        args.memmap_path = config.get("MEMMAP_PATH")
        
    # CHECK_POINT_FILE = 'mil_checkpoint_draft.pth' 
    CHECK_POINT_FILE = 'mil_checkpoint.pth'   
    main(args) 
   
   
   
 