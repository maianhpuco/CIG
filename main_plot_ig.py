import os
import sys 
import torch
from tqdm import tqdm 
import glob
import pandas as pd
import numpy as np
import argparse
import time   
import shutil 
import h5py
import random 
import numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader 
from utils.utils import load_config
from utils.plotting import (
    plot_heatmap_with_bboxes,
    get_region_original_size,
    downscaling,
    rescaling_stat_for_segmentation, 
    min_max_scale, 
    replace_outliers_with_bounds 
) 
from datasets.ig_dataset import IG_dataset 
import openslide
import glob 
import matplotlib.pyplot as plt
 
def main(args): 
    '''
    Input: h5 file
    Output: save scores into a json folder
    '''
    all_scores_paths = glob.glob(os.path.join(args.attribution_scores_dir, f'{args.ig_name}', "*.npy"))
        
    plots_dir = os.path.join(args.plots_dir, f'{args.ig_name}')    
    
    if os.path.exists(plots_dir):
        shutil.rmtree(plots_dir)  # Delete the existing directory
    os.makedirs(plots_dir)  
        
    for idx, scores_path in enumerate(all_scores_paths):
        print(f"Print the plot {idx+1}/{len(all_scores_paths)}")
        print(scores_path)
        scores_array = np.load(scores_path)
        print("scores array shape", scores_array.shape)
        basename = os.path.basename(scores_path).split(".")[0]
        if basename not in ['normal_001']:
            continue
        slide = openslide.open_slide(os.path.join(args.slides_dir, f'{basename}.tif'))
        (
            downsample_factor,
            new_width,
            new_height,
            original_width,
            original_height
        ) = rescaling_stat_for_segmentation(
            slide, downsampling_size=1096)

        scale_x = new_width / original_width
        scale_y = new_height / original_height
        h5_file_path = os.path.join(args.features_h5_dir, f'{basename}.h5') 
        
        result = {} 
        with h5py.File(h5_file_path, "r") as f:
            coordinates= f['coordinates'][:]
        scaled_scores = min_max_scale(replace_outliers_with_bounds(scores_array.copy()))
        
        plot_path = os.path.join(plots_dir, f'{basename}.png')
        plot_heatmap_with_bboxes(
            scale_x, scale_y, new_height, new_width,
            coordinates,
            scaled_scores,
            name = "",
            save_path = plot_path
        ) 
        print("-> Save the plot at: ", plot_path)
        # scale_x, scale_y, new_height, new_width  


if __name__ == '__main__':
    # get config 
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
    
    args = parser.parse_args()
    
    if not os.path.exists(f'./{args.config_file}'):
        raise ValueError(f"{args.config_file} does not exist")
    
    config = load_config(f'{args.config_file}')
    args.use_features = config.get('USE_FEATURES', True)  
    args.slides_dir = config.get('SLIDES_DIR')
    args.plots_dir = config.get('PLOTS_DIR')
    args.feature_mean_std_path = config.get('FEATURE_MEAN_STD_PATH')
    args.patch_path = config.get('PATCH_PATH') # save all the patch (image)
    args.features_h5_dir = config.get("FEATURES_H5_DIR") # save all the features
    args.checkpoints_dir = config.get("CHECKPOINTS_DIR")
    args.attribution_scores_dir = config.get("ATTRIBUTION_SCORES_DIR")    
    args.plot_dir = config.get("PLOTS_DIR")
    os.makedirs(args.attribution_scores_dir, exist_ok=True) 
    os.makedirs(args.plot_dir, exist_ok=True) 
    args.batch_size = config.get('BATCH_SIZE')
    args.feature_extraction_model = config.get('FEATURE_EXTRACTION_MODEL')
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    main(args)