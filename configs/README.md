# Hydra Configuration Usage Guide

## Overview
Your training script now uses Hydra for configuration management. All hyperparameters are stored in YAML files under `configs/`.

## Directory Structure
```
configs/
├── config.yaml              # Main config (defaults)
├── model/
│   ├── baseline.yaml        # Baseline CNN config
│   ├── alexnet.yaml         # AlexNet config
│   └── vgg16.yaml           # VGG16 config
└── training/
    ├── default.yaml         # Default training params
    └── quick.yaml           # Quick test params
```

## Basic Usage

### 1. Train with defaults (baseline model, default training)
```bash
python -m main_project.train
```

### 2. Train VGG16 model
```bash
python -m main_project.train model=vgg16
```

### 3. Train AlexNet model
```bash
python -m main_project.train model=alexnet
```

### 4. Quick test run (3 epochs)
```bash
python -m main_project.train training=quick
```

### 5. Combine configurations
```bash
# VGG16 with quick training
python -m main_project.train model=vgg16 training=quick
```

## Override Parameters

You can override any parameter from the command line:

```bash
# Change learning rate
python -m main_project.train training.lr=0.0001

# Change batch size and epochs
python -m main_project.train training.batch_size=64 training.epochs=20

# Change patience for early stopping
python -m main_project.train training.patience=10

# Force CPU training
python -m main_project.train device=cpu

# Multiple overrides
python -m main_project.train model=vgg16 training.lr=0.0001 training.batch_size=64 training.epochs=50
```

## Configuration Files

### Main Config (`config.yaml`)
- Sets default model and training configs
- Defines data paths and device settings
- Specifies output directories

### Model Configs (`model/*.yaml`)
- `name`: Model architecture (baseline, alexnet, vgg16)
- `num_classes`: Number of output classes
- `pretrained`: Whether to use pretrained weights

### Training Configs (`training/*.yaml`)
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `lr`: Learning rate
- `patience`: Early stopping patience

## Hydra Features

### Automatic Output Directories
Hydra creates timestamped output directories for each run:
```
outputs/
└── 2026-01-11/
    └── 18-45-30/
        ├── .hydra/          # Config snapshots
        ├── train.log        # Logs
        └── ...
```

### View Configuration
See the full configuration being used:
```bash
python -m main_project.train --cfg job
```

### Help
```bash
python -m main_project.train --help
```

## Examples

### Experiment 1: Compare models
```bash
# Baseline
python -m main_project.train model=baseline

# AlexNet
python -m main_project.train model=alexnet

# VGG16
python -m main_project.train model=vgg16
```

### Experiment 2: Learning rate sweep
```bash
python -m main_project.train training.lr=0.1
python -m main_project.train training.lr=0.01
python -m main_project.train training.lr=0.001
python -m main_project.train training.lr=0.0001
```

### Experiment 3: Quick debugging
```bash
python -m main_project.train training=quick model=baseline
```

## Tips

1. **Create custom configs**: Add new YAML files in `configs/model/` or `configs/training/` for your experiments
2. **Config groups**: Use `model=` and `training=` to switch between config groups
3. **Overrides**: Use dot notation (`training.lr=0.001`) to override specific values
4. **Reproducibility**: Hydra saves all configs used for each run in `outputs/`
