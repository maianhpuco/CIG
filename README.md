# CIG (Contrastive Integrated Gradients)

A Python-based framework for computing and analyzing various gradient-based attribution methods for Multiple Instance Learning (MIL) models in medical imaging.

## Overview

This project implements Contrastive Integrated Gradients (CIG), a novel attribution method for Multiple Instance Learning (MIL) models in medical imaging. CIG extends traditional Integrated Gradients by incorporating contrastive learning principles to better highlight discriminative regions in medical images.

Key features:
- Implementation of Contrastive Integrated Gradients for MIL models
- Support for whole slide image (WSI) analysis
- Visualization tools for attribution maps
- Configurable architecture for different feature extractors and MIL aggregators
- Efficient batch processing of large medical images

The method helps identify regions in medical images that are most relevant for model predictions by comparing and contrasting features between different classes. This is particularly useful for tasks like cancer detection where understanding which image regions influence the model's decision is critical.


## Project Structure

```
.
├── src/                        # Source code
├── main_ig.py                  # Main script for attribution computation
├── main_plot_ig.py             # Script for plotting attribution results
└── config.yaml.example         # Example configuration file
```

## Requirements

- Anaconda (https://www.anaconda.com/docs/getting-started/anaconda/install)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/maianhpuco/CIG.git
cd CIG
```

2. Create and activate a new conda environment:
```bash
conda env create -f environment.yml
```

## Usage

### Computing Attribution Scores

To compute attribution scores using a specific method:

```bash
python main_ig.py --config_file [config_name] --bag_classifier [model]
python main_plot_ig.py --config_file [config_name]
```

### Configuration
1. Download all folders needed from link https://drive.google.com/drive/folders/1tgff35Qx2CpvW1YUfPoWtL820tdVbZ4X?usp=drive_link

2. Create a new configuration file by copying the example:
```bash
cp config.yaml.example config.yal
```

The project uses a YAML configuration file to specify:
- Feature directories
- Checkpoint locations
- Model parameters
- Data paths

Example configuration:
```yaml
SLIDES_DIR: "path/to/slides"
FEATURES_H5_DIR: "path/to/features"
CHECKPOINTS_DIR: "path/to/checkpoints"
ATTRIBUTION_SCORES_DIR: "path/to/scores"
```

## Model Support

The framework supports multiple MIL model architectures:
- MIL (Multiple Instance Learning)
- CLAM (Clustering-constrained Attention Multiple instance learning)
- DSMIL (Dual-stream Multiple Instance Learning)