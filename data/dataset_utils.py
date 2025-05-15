import math
import numpy as np
from torchvision import transforms
from skimage.transform import resize 
import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries 

from skimage import segmentation
from skimage import color
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2
 

def superpixel_segmenting(obj, downsample_size = 1096, n_segments=2000, compactness=10.0, start_label=0):
    downsample_factor, new_width, new_height, curr_width, curr_height = rescaling_stat_for_segmentation(
        obj, downsample_size)

    # Downscale the region and prepare for mask generation
    downscaled_region = downscaling(
        obj, new_width, new_height)
    downscaled_region_array = np.array(downscaled_region)

    lab_image = color.rgb2lab(downscaled_region_array)
    superpixel_labels = segmentation.slic(
        lab_image, n_segments=n_segments, compactness=compactness, start_label=start_label)

    # print((time.time()-start)/60.00)
    segmented_mask = segmentation.mark_boundaries(
        downscaled_region_array, superpixel_labels)

    return superpixel_labels, segmented_mask, downsample_factor, new_width, new_height, downscaled_region_array, lab_image  

 
def rescaling_stat_for_segmentation(obj, downsampling_size=1024):
    """
    Rescale the image to a new size and return the downsampling factor.
    """
    if hasattr(obj, 'shape'):
        original_width, original_height = obj.shape[:2]
    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        original_width, original_height = obj.size
    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        original_width, original_height = obj.dimensions
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

    if original_width > original_height:
        downsample_factor = int(downsampling_size * 100000 / original_width) / 100000
    else:
        downsample_factor = int(downsampling_size * 100000 / original_height) / 100000

    new_width = int(original_width * downsample_factor)
    new_height = int(original_height * downsample_factor)

    return downsample_factor, new_width, new_height, original_width, original_height


def downscaling(obj, new_width, new_height):
    """
    Downscale the given object (image or slide) to the specified size.
    """
    if isinstance(obj, np.ndarray):  # If it's a NumPy array
        # Resize using scikit-image (resize scales and interpolates)
        image_numpy = resize(obj, (new_height, new_width), anti_aliasing=True)
        image_numpy = (image_numpy * 255).astype(np.uint8)

    elif hasattr(obj, 'size'):  # If it's an image (PIL or similar)
        obj = obj.resize((new_width, new_height))
        image_numpy = np.array(obj)

    elif hasattr(obj, 'dimensions'):  # If it's a slide (e.g., a TIFF object)
        thumbnail = obj.get_thumbnail((new_width, new_height))
        transform = transforms.Compose([
            transforms.ToTensor(),  # Converts the image to a tensor (C, H, W)
        ])
        image_tensor = transform(thumbnail)
        image_numpy = image_tensor.permute(1, 2, 0).numpy()  # Convert from (C, H, W) to (H, W, C) numpy format
    else:
        raise ValueError("The object must have either 'size' (image) or 'dimensions' (slide) attribute.")

    return image_numpy 

def equalize_image(downscaled_region_array):
    # Convert to YUV color space
    yuv_image = cv2.cvtColor(downscaled_region_array, cv2.COLOR_RGB2YUV)

    # Extract the Y (luminance) channel
    y_channel = yuv_image[..., 0]

    # Ensure the Y channel is 8-bit single channel (CV_8UC1)
    y_channel = np.uint8(y_channel)  # Convert to 8-bit unsigned integer

    # Apply histogram equalization to the Y channel (luminance)
    y_channel_equalized = cv2.equalizeHist(y_channel)

    # Replace the Y channel in the YUV image with the equalized Y channel
    yuv_image[..., 0] = y_channel_equalized

    # Convert back to RGB color space
    equalized_image = cv2.cvtColor(yuv_image, cv2.COLOR_YUV2RGB)

    return equalized_image
 
def identify_foreground_background(equalized_image, superpixel_labels, threshold=240):
    """
    """
    # Convert to a numpy array and initialize the lists
    equalized_image = np.array(equalized_image)
    unique_superpixels = np.unique(superpixel_labels)

    # Determine the pixels that are close to white (background pixels)
    background_mask = np.all(equalized_image >= threshold, axis=-1)  # RGB close to white

    foreground_superpixels = []
    background_superpixels = []

    # Iterate through each superpixel label
    for label in unique_superpixels:
        # Create a mask for the current superpixel
        superpixel_mask = superpixel_labels == label

        # Check the percentage of background pixels within this superpixel
        superpixel_background = np.sum(background_mask[superpixel_mask]) / np.sum(superpixel_mask)

        # If more than 50% of the superpixel is background (white), consider it background
        if superpixel_background > 0.5:
            background_superpixels.append(label)
        else:
            foreground_superpixels.append(label)

    return foreground_superpixels, background_superpixels