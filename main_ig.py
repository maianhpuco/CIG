import os
import torch
import numpy as np
import argparse
import time   
import numpy as np
from src.bag_classifier.mil_classifier import MILClassifier 
import torch
import torch.optim as optim
from src.bag_classifier.mil_classifier import MILClassifier
from src.utils import load_config
from src.utils.train_classifier.train_mlclassifier import load_checkpoint
import h5py
from src.datasets.ig_dataset import IG_dataset 
from src.attr_method._common import sample_random_features, call_model_function
from src.attr_method.contrastive_gradient import ContrastiveGradients as AttrMethod 

def load_model(checkpoint_path):
    input_dim = 768  # Adjust according to dataset
    mil_model = MILClassifier(input_dim=input_dim, pooling='attention')
    optimizer = optim.AdamW(mil_model.parameters(), lr=0.0005)
    model, _, _, _ = load_checkpoint(mil_model, optimizer, checkpoint_path)  
    return model 

def main(args): 
    #----------------------------------------------------    
    attribution_method = AttrMethod()   
    
    score_save_path = os.path.join(args.attribution_scores_dir, 'contrastive_gradient') 

    checkpoint_path = os.path.join(args.checkpoints_dir, f'{CHECK_POINT_FILE}')
    mil_model = load_model(checkpoint_path)
    
    basenames = [] 
    for basename in os.listdir(args.slides_dir):
        basename = basename.split(".")[0]
        if basename.startswith('normal_'): 
            basenames.append(basename)
    
    dataset = IG_dataset(
        args.features_h5_dir,
        args.slides_dir,
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
        stacked_features_baseline, selected_basenames = sample_random_features(dataset, num_files=20) 
        stacked_features_baseline = stacked_features_baseline.numpy() 
        
        kwargs = {
            "x_value": features,  
            "call_model_function": call_model_function,  
            "model": mil_model,  
            "baseline_features": stacked_features_baseline,  # Optional
            "memmap_path": args.memmap_path, 
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
    parser.add_argument('--config_file', default='config.yaml')
    parser.add_argument('--bag_classifier', 
                    type=str, 
                    default='mil', 
                    choices=[
                        'mil', 
                        'clam', 
                        'dsmil'
                    ],
                    help='Choose the bag classifier to use.')
    args = parser.parse_args()

    if not os.path.exists(f'./{args.config_file}'):
        raise ValueError(f"{args.config_file} does not exist")
    
    config = load_config(f'./{args.config_file}')
    args.use_features = config.get('USE_FEATURES', True)
    args.slides_dir = config.get('SLIDES_DIR')
    args.features_h5_dir = config.get("FEATURES_H5_DIR") # save all the features
    args.checkpoints_dir = config.get("CHECKPOINTS_DIR")
    args.attribution_scores_dir = config.get("ATTRIBUTION_SCORES_DIR")    
    args.plots_dir = config.get("PLOTS_DIR") 
    os.makedirs(args.features_h5_dir, exist_ok=True)  
    os.makedirs(args.attribution_scores_dir, exist_ok=True) 
    args.batch_size = config.get('BATCH_SIZE')
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.feature_mean_std_path=config.get("FEATURE_MEAN_STD_PATH")
    args.do_normalizing = True
    args.memmap_path = config.get("MEMMAP_PATH")
    
    if args.bag_classifier=='mil':
        CHECK_POINT_FILE = 'mil_checkpoint.pth'   
    elif args.bag_classifier=='clam':
        CHECK_POINT_FILE = 'clam_checkpoint.pth'
    elif args.bag_classifier=='dsmil':
        CHECK_POINT_FILE = 'dsmil_checkpoint.pth'
    else:
        raise ValueError(f"Invalid bag classifier: {args.bag_classifier}")
    
    main(args)