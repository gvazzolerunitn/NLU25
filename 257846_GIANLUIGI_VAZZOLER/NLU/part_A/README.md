# Natural Language Understanding - Part A

## Overview

This code implements LSTM-based models for joint Intent Classification and Slot Filling on the ATIS dataset. The models simultaneously predict user intents (e.g., "flight_search") and extract slot entities (e.g., city names, dates) from natural language utterances. It supports both unidirectional and bidirectional LSTM architectures with configurable dropout regularization.

## Features

- Joint intent classification and slot filling
- LSTM-based sequence modeling with bidirectional support
- Dropout regularization for improved generalization
- CoNLL evaluation metrics for slot filling (F1-score)
- Classification report for intent recognition
- Multiple experimental configurations
- Statistical analysis with multiple runs
- Comprehensive result tracking
- Early stopping with patience mechanism

## Project Structure

```
part_A/
├── main.py          # Experiment configurations and execution
├── functions.py     # Training/evaluation loops and pipeline
├── model.py         # LSTM model architecture (ModelIAS)
├── utils.py         # Data loading, Lang class, and preprocessing
├── conll.py         # CoNLL evaluation metrics
└── dataset/ATIS/    # ATIS dataset files
```

## Training

Configure experiments in `main.py` by modifying the `experiments_config` dictionary:

```python
experiments_config = {
    "LSTM_dropout0.3_bidir": {
        "dropout": 0.3,
        "bidirectional": True,
        "run": True,         # Train new model
        "n_runs": 3          # Number of runs for statistical significance
    }
}
```

Parameters:
- `dropout`: Dropout rate (0.0-0.5)
- `bidirectional`: Enable bidirectional LSTM
- `run`: Set to `True` to train from scratch
- `n_runs`: Number of training runs for averaging results

Run training:
```bash
cd part_A
python main.py
```

## Evaluating Pre-trained Models

To evaluate an existing model without retraining:

1. **Set the experiment configuration**:
```python
experiments_config = {
    "LSTM_dropout0.3_bidir_best": {  # Must match saved model name
        "dropout": 0.3,
        "bidirectional": True,
        "run": False,        # Skip training
        "n_runs": 3          # Run evaluation multiple times
    }
}
```

2. **Ensure the model file exists**:
   - Model must be saved at: `model_bin/LSTM_dropout0.3_bidir_best.pt`

3. **Run evaluation**:
```bash
python main.py
```

**Expected output**:
```
Running experiment LSTM_dropout0.3_bidir_best
Slot F1: 0.944 ± 0.000
Intent Acc: 0.951 ± 0.000
```

**Note**: The `± 0.000` is normal for evaluation since it's deterministic (no randomness).

## Model Architecture

The `ModelIAS` class implements:
- Embedding layer for word representations
- LSTM encoder (uni/bidirectional) with dropout
- Dual output heads for slots and intents
- Sequence-to-sequence slot prediction
- Last hidden state for intent classification

## Output

Training produces:
- **Models**: Saved in `model_bin/experiment_name.pt` with model state and vocabulary
- **Results**: Test set performance with statistical summaries
- **Logs**: Training progress printed to console

## Data Format

The ATIS dataset uses:
- **Utterances**: Tokenized user queries
- **Slots**: BIO-tagged slot labels
- **Intents**: Single intent per utterance

Example:
```
Utterance: ["show", "me", "flights", "from", "boston"]
Slots:     ["O",    "O",  "O",       "O",    "B-city"]
Intent:    "flight_search"
```

## Requirements

- PyTorch
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
