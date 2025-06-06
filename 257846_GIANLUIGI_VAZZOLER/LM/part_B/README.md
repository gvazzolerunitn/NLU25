# Language Modeling - Part B

## Overview

This code implements modifications to the baseline LSTM architecture including:
- **Weight Tying**: Shared embeddings between input and output layers
- **Variational (Locked) Dropout**: Consistent dropout masks across time steps
- **NT-AvSGD**: Non-monotonic Averaged Stochastic Gradient Descent for better convergence

The models are trained on Penn TreeBank dataset and support both training and evaluation modes.

## Features

- Advanced LSTM with weight tying
- Variational dropout implementation
- NT-AvSGD optimization strategy
- Standard early stopping fallback
- Comprehensive experiment tracking
- Automatic model selection and averaging
- Training visualization and logging

## Project Structure

```
part_B/
├── main.py          # Main script with experiment configurations
├── functions.py     # Training loops and utility functions
├── model.py         # Advanced LSTM model with variational dropout
├── utils.py         # Data loading and preprocessing
└── dataset/         # Penn TreeBank dataset
```

## Training

The training is configured through the `EXPERIMENTS` list in `main.py`. Each experiment defines:

```python
{
    "name": "LSTM_WT_VD_NTAvSGD4",
    "use_vdropout": True,      # Enable variational dropout
    "use_ntasgd": True,        # Enable NT-AvSGD averaging
    "lr": 4.0,
    "emb_size": 450,
    "hid_size": 450,
    "batch_size": 64,
    "dropout_rate": 0.4
}
```

To run training:
```bash
python main.py --mode train
```

This will run all experiments defined in the `EXPERIMENTS` list.

## Evaluation

To evaluate a specific pre-trained model:

```bash
python main.py --mode eval --eval_exp "experiment_name" --model_dir "model_bin/experiment_folder"
```

Parameters:
- `--eval_exp`: Name of the experiment configuration to use
- `--model_dir`: Path to the folder containing `model.pt`

The evaluation script will:
1. Load the specified model configuration
2. Rebuild the model architecture
3. Load the saved weights
4. Evaluate on the test set and report perplexity

## Key Techniques

### Weight Tying
Input embeddings and output projection share the same weight matrix, reducing parameters and improving generalization.

### Variational Dropout
Uses consistent dropout masks across sequence time steps, providing more stable regularization than standard dropout.

### NT-AvSGD
Monitors validation performance and automatically starts parameter averaging when validation stops improving, leading to better final models.

## Output

Training produces:
- **Model checkpoints**: Saved in `model_bin/experiment_name/model.pt`
- **Training logs**: Saved in `runs/experiment_name/training_log.csv`
- **Training curves**: Saved in `runs/experiment_name/training_curves.png`

The logs include:
- Training/validation loss per epoch
- Validation perplexity progression
- Final test perplexity
- Configuration parameters

## Requirements

- PyTorch
- matplotlib
- numpy
- tqdm
- csv

## Common Parameters

- `clip`: Gradient clipping threshold (default: 5)
- `n_epochs`: Maximum training epochs (default: 100)
- `patience`: Early stopping patience (default: 5)
- `ntasgd_trigger`: Epochs to wait before NT-AvSGD trigger (default: 5)
