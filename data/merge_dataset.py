import torch
from torch.utils.data import Dataset
import h5py
import os
import sys 
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import glob
import openslide 
import json 
import cv2 
import time 
from PIL import Image 
from scipy import ndimage
from PIL import Image, ImageFilter
from PIL import ImageStat 

class SlidePatchesDataset(Dataset):
    """Read all the Patches within the slide"""
    def __init__(self, patch_dir, transform):
        self.patch_dir = patch_dir
        self.transform = transform
        self.patch_files = [os.path.join(patch_dir, f) for f in os.listdir(patch_dir) if f.endswith('.png')]
    
    def __len__(self):
        return len(self.patch_files)
    
    @staticmethod
    def parse_patch_name(patch_filename):
        # Remove the file extension
        patch_name = os.path.splitext(patch_filename)[0]
        
        # Split the patch name by underscores
        parts = patch_name.split('_')
        
        # Extract values from the parts list and return as a dictionary
        return {
            'ymin': int(parts[0]),
            'ymax': int(parts[1]),
            'xmin': int(parts[2]),
            'xmax': int(parts[3]),
            'spixel_idx': int(parts[4]),
            'patch_idx': int(parts[5])
        }
     
    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        patch_image = Image.open(patch_path).convert('RGB')  # Open and convert to RGB
        patch_name = os.path.basename(patch_path) 
        
        patch_info = self.parse_patch_name(patch_name)
        if self.transform:
            patch_image = self.transform(patch_image)  # Apply transformations if any

        return {'image': patch_image, 'patch_info': patch_info} 

class SuperpixelPatchesDataset(Dataset):
    """Read all the patches within the slide for a specific superpixel"""
    def __init__(self, patch_dir, transform, preferred_spixel_idx):
        """
        Args:
            patch_dir (str): Directory where patches are saved.
            transform (callable, optional): Optional transform to be applied on a sample.
            preferred_spixel_idx (int): The superpixel index that you want to filter patches by.
        """
        self.patch_dir = patch_dir
        self.transform = transform
        self.preferred_spixel_idx = preferred_spixel_idx

        # Only load patches whose spixel_idx matches preferred_spixel_idx
        self.patch_files = [
            os.path.join(patch_dir, f) for f in os.listdir(patch_dir) 
            if f.endswith('.png') and self._is_matching_spixel(f)
        ]

         
    def __len__(self):
        return len(self.patch_files)
    
    def _is_matching_spixel(self, patch_filename):
        """Check if the patch's spixel_idx matches the preferred spixel_idx"""
        patch_info = self.parse_patch_name(patch_filename)
        return patch_info['spixel_idx'] == self.preferred_spixel_idx
    
    @staticmethod
    def parse_patch_name(patch_filename):
        """Parse the patch filename to extract the bounding box and other information"""
        patch_name = os.path.splitext(patch_filename)[0]
        parts = patch_name.split('_')
        
        return {
            'ymin': int(parts[0]),
            'ymax': int(parts[1]),
            'xmin': int(parts[2]),
            'xmax': int(parts[3]),
            'spixel_idx': int(parts[4]),
            'patch_idx': int(parts[5])
        }
    
    def __getitem__(self, idx):
        patch_path = self.patch_files[idx]
        patch_image = Image.open(patch_path).convert('RGB')  # Open and convert to RGB
        patch_name = os.path.basename(patch_path)
        
        patch_info = self.parse_patch_name(patch_name)
        
        if self.transform:
            patch_image = self.transform(patch_image)  # Apply transformations if any
        
        return {'image': patch_image, 'patch_info': patch_info}

class PatchDataset(Dataset):
    def __init__(
        self,
        region,
        mask,
        patch_size=(224, 224),
        coverage_threshold=0.1,
        edge_threshold = 15, #similar to Camil method 
        transform=None,
        return_feature=False,
        model=None
        ):
        self.region_np = region
        self.mask = mask
        self.patch_size = patch_size
        self.coverage_threshold = coverage_threshold
        self.edge_threshold = edge_threshold
        self.transform = transform
        self.model = model
        self.return_feature = return_feature

        # Get region dimensions and patch size
        region_height, region_width = region.shape[:2]
        patch_height, patch_width = patch_size

        self.patches = []
        self.bboxes = []
        self.patch_indices = []  # Keep track of original patch indices
        self.patch_idx_dict = {}
        # Loop through the region and extract patches
        patch_original_idx = 0  # Initialize the patch index
        idx = 0
        
        for top in range(0, region_height, patch_height):
            for left in range(0, region_width, patch_width):
                # Ensure the patch is within bounds
                bottom = min(top + patch_height, region_height)
                right = min(left + patch_width, region_width)

                # Extract the patch and corresponding mask region
                patch = region[top:bottom, left:right]
                patch_mask = mask[top:bottom, left:right]

                patch_area = patch.shape[0] * patch.shape[1]
                mask_coverage = np.sum(patch_mask) / patch_area  # Proportion of the patch covered by the mask
                
                # Only include patches that satisfy the coverage threshold
                if mask_coverage > self.coverage_threshold :
                    edge_mean = self.filter_by_edge_detection(
                        patch, 
                        patch_area
                    ) 
                    if edge_mean > self.edge_threshold: 
                        bbox = (top, left, bottom, right)
                        self.patches.append(patch)
                        self.bboxes.append(bbox)
                        self.patch_indices.append(patch_original_idx)  # Save the original index for each patch

                        _idx_dict = {idx: patch_original_idx}
                        self.patch_idx_dict.update(_idx_dict)
                        idx += 1        
                patch_original_idx += 1  # Increment the original index after processing each patch

    @staticmethod
    def filter_by_edge_detection(patch, patch_area):
        # Convert the NumPy array (patch) to a PIL image
        patch_pil = Image.fromarray(patch.astype(np.uint8))

        # Apply edge detection using PIL's ImageFilter.FIND_EDGES
        edge = patch_pil.filter(ImageFilter.FIND_EDGES)

        # Compute the sum of the edge values using ImageStat
        edge_stat = ImageStat.Stat(edge).sum
        edge_mean = np.mean(edge_stat) / patch_area  # Normalize by patch area

        return edge_mean 
    
    def __len__(self):
        """Returns the total number of patches."""
        return len(self.patches)

    def __getitem__(self, idx):
        """Returns a patch, its corresponding bounding box and its original index."""
        patch = self.patches[idx]
        bbox = self.bboxes[idx]
        patch_idx = self.patch_idx_dict[idx]  # Get the original index for the patch
        # Convert patch to PIL image for torchvision transforms
        patch_pil = Image.fromarray(patch.astype(np.uint8))  # Convert numpy array to PIL Image

        # Apply the transformations if provided
        if self.transform:
            patch_pil = self.transform(patch_pil)

        if self.return_feature:
            patch_tensor = patch_pil.unsqueeze(0)  # Add batch dimension
            with torch.no_grad():
                features = self.model.forward_features(patch_tensor)
            class_token_features = features[:, 0, :]
            return class_token_features.squeeze(0), patch_pil, bbox, patch_idx  # Return original index
        else:
            return _, patch_pil, bbox, patch_idx  # Return original index 

class SuperpixelDataset(Dataset):
    def __init__(self, slide_paths, json_folder):
        """
        Args:
        """
        self.slide_paths = slide_paths
        self.json_folder = json_folder

        # Get list of WSI paths and filter by example_list

    def __len__(self):
        """Returns the total number of samples (WSI images)."""
        return len(self.slide_paths)

    def __getitem__(self, index):
        """Returns a sample (WSI image and associated data)."""
        # Get the WSI path and basename
        wsi_path = self.slide_paths[index]
        basename = os.path.basename(wsi_path).split(".")[0]

        # slide = openslide.open_slide(wsi_path)
        # Load corresponding JSON data
        json_path = os.path.join(self.json_folder, f'{basename}.json')
     
        ######################################## 
        wt_json_path = os.path.join(self.json_folder, f'{basename}')
        if not os.path.exists(json_path) and os.path.exists(wt_json_path):
            # If wt_json_path exists, rename it to json_path (with .json extension)
            os.rename(wt_json_path, json_path)
            print(f"Renamed {wt_json_path} to {json_path}")
    
        # Now that json_path should exist, read the file
        if os.path.exists(json_path):
            sample = self.read_json_superpixel(json_path)
        else:
            raise FileNotFoundError(f"JSON file for {basename} not found.")
        ########################################  
        sample = self.read_json_superpixel(json_path)
        
        bounding_boxes = sample['bounding_boxes']
        downsample_factor = sample['downsample_factor']
        foreground_superpixels = sample['foreground_superpixels']
        superpixel_labels = sample['superpixel_labels']

        result = [] 
        
        for foreground_idx in foreground_superpixels:
            bbox = bounding_boxes[foreground_idx]
            xywh_abs_bbox = self._get_absolute_bbox_coordinate(bbox, downsample_factor)

            superpixel_downsampling = superpixel_labels == foreground_idx
            superpixel_extrapolated = self.extrapolate_superpixel_mask_segment(
                superpixel_labels, foreground_idx, bounding_boxes, downsample_factor)
            result.append({
                        'foreground_idx': foreground_idx,
                        'xywh_abs_bbox': xywh_abs_bbox,
                        'superpixel_extrapolated': superpixel_extrapolated
                    })
        return result, wsi_path # can also read and return the numpy file of the region_np after segmentation 
    
    @staticmethod
    def extrapolate_superpixel_mask_segment(
        superpixel_downsampling,
        superpixel_idx,
        bounding_boxes,
        downsample_factor):
        mask = (superpixel_downsampling == superpixel_idx).astype(np.uint8)
        xmin, ymin, xmax, ymax = [int(i) for i in bounding_boxes[superpixel_idx]]
        cropped_mask = mask[ymin:ymax, xmin:xmax]  # Corrected cropping

        upscaled_mask = cv2.resize(
            cropped_mask,
            (int(cropped_mask.shape[1] / downsample_factor), int(cropped_mask.shape[0] / downsample_factor)),
            interpolation=cv2.INTER_NEAREST  # Nearest-neighbor interpolation to keep binary mask intact
        )

        upscaled_mask_bool = (upscaled_mask > 0).astype(bool)  # Convert to boolean (True/False)

        return upscaled_mask_bool

    @staticmethod
    def _get_absolute_bbox_coordinate(bbox, downsample_factor):
        xmin, ymin, xmax, ymax = bbox
        width = xmax - xmin
        height = ymax - ymin

        xmin_original = int(xmin / downsample_factor)
        ymin_original = int(ymin / downsample_factor)
        width_original = int(width / downsample_factor)
        height_original = int(height / downsample_factor)

        relative_bbox = [xmin_original, ymin_original, width_original, height_original]

        return relative_bbox

    @staticmethod
    def read_json_superpixel(json_path):
        with open(json_path, 'r') as json_file:
            loaded_data = json.load(json_file)

        # Process JSON data
        superpixel_labels = np.array(loaded_data['superpixel_labels'])
        downscaled_region_array = np.array(loaded_data['downscaled_region_array'])
        output_image_with_bboxes = np.array(loaded_data['output_image_with_bboxes'])

        foreground_superpixels = [int(i) for i in loaded_data['foreground_superpixels']]
        background_superpixels = [int(i) for i in loaded_data['background_superpixels']]

        bounding_boxes = {int(k): tuple(v) for k, v in loaded_data['bounding_boxes'].items()}

        downsample_factor = loaded_data['downsample_factor']
        new_width = loaded_data['new_width']
        new_height = loaded_data['new_height']

        # Return the sample (image, features, bounding boxes)
        sample = {
            'superpixel_labels': superpixel_labels,
            'downscaled_region_array': downscaled_region_array,
            'output_image_with_bboxes': output_image_with_bboxes,
            'foreground_superpixels': foreground_superpixels,
            'background_superpixels': background_superpixels,
            'bounding_boxes': bounding_boxes,
            'downsample_factor': downsample_factor,
            'new_width': new_width,
            'new_height': new_height
        }

        return sample 