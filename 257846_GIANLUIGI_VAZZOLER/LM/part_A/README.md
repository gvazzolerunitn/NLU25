# Language Modeling - Part A

## Overview

This code implements LSTM-based language models for the Penn TreeBank dataset. It includes basic LSTM architectures with dropout regularization and supports both SGD and AdamW optimizers. The models are trained to predict the next word in a sequence, with performance measured by perplexity.

## Features

- LSTM language model implementation
- Dropout regularization support
- Multiple optimizer options (SGD, AdamW)
- Early stopping with patience
- Automatic model checkpointing
- Training curve visualization
- Comprehensive logging (CSV format)

## Project Structure

```
part_A/
├── main.py          # Main training/evaluation script
├── functions.py     # Training loops and utility functions
├── model.py         # LSTM model architecture
├── utils.py         # Data loading and preprocessing
└── dataset/         # Penn TreeBank dataset
```

## Training

To train a model, modify the configuration in `main.py`:

```python
config = {
    "mode": "train",
    "name": "my_experiment",
    "optimizer": "SGD",        # or "AdamW"
    "emb_size": 300,
    "hid_size": 200,
    "lr": 1.0,
    "batch_size": 32,
    "dropout_rate": 0.4,
    "clip": 5,
    "n_epochs": 100,
    "patience": 3
}
```

Then run:
```bash
python main.py
```

## Evaluation

To evaluate a pre-trained model:

1. Set the configuration mode to "eval":
```python
config = {
    "mode": "eval",
    "model_dir": "model_bin/your_experiment_name",
    # ... other parameters should match training config
}
```

2. Run evaluation:
```bash
python main.py
```

The script will load the best model checkpoint (`model_best.pt`) and evaluate it on the test set.

## Output

Training produces:
- **Model checkpoints**: Saved in `model_bin/experiment_name/`
  - `model.pt`: Final epoch model
  - `model_best.pt`: Best validation perplexity model
- **Training logs**: Saved in `runs/experiment_name/training_log.csv`
- **Training curves**: Saved in `runs/experiment_name/training_curves.png`

## Requirements

- PyTorch
- matplotlib
- numpy
- tqdm
