# CIG (Contrastive Integrated Gradients)

A Python-based framework for computing and analyzing various gradient-based attribution methods for Multiple Instance Learning (MIL) models in medical imaging.

## Overview

This project implements and compares different gradient-based attribution methods for Multiple Instance Learning models, specifically designed for medical image analysis. It supports various attribution techniques including:

- Integrated Gradients
- Vanilla Gradients
- Contrastive Gradients
- Expected Gradients
- Integrated Decision Gradients
- Square Integrated Gradients
- Optimized Square Integrated Gradients

## Project Structure

```
.
├── attr_method/                # Attribution method implementations
├── datasets/                   # Dataset handling code
├── src/                        # Source code
├── utils/                      # Utility functions
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
python main_ig.py --ig_name [method_name] --config_file [config_name] --bag_classifier [model]
```

Available attribution methods:
- `integrated_gradient`
- `vanilla_gradient`
- `contrastive_gradient`
- `expected_gradient`
- `integrated_decision_gradient`
- `square_integrated_gradient`
- `optim_square_integrated_gradient`

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