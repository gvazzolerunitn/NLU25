# Natural Language Understanding - Part B

## Overview

This code implements BERT-based models for joint Intent Classification and Slot Filling using the ATIS dataset. It leverages pre-trained BERT (base or large) with custom classification heads for simultaneous intent recognition and slot entity extraction.

## Features

- BERT-based joint NLU with dual prediction heads
- Support for both BERT-base and BERT-large models
- Proper subword tokenization and label alignment
- Adam optimizer with configurable learning rates
- Early stopping based on slot F1-score
- Multiple runs for statistical significance
- Comprehensive evaluation metrics

## Project Structure

```
part_B/
├── main.py          # Experiment configurations
├── functions.py     # Training/evaluation loops and experiment runner
├── model.py         # BERT-based joint NLU model (JointBertForNLU)
├── utils.py         # BERT tokenization, dataset, and data utilities
├── conll.py         # CoNLL evaluation metrics
└── dataset/ATIS/    # ATIS dataset files
```

## Training

Configure experiments in `main.py`:

```python
to_run = {
    "BertBase_Baseline": {
        "run": True,
        "batch_size": 16,
        "lr": 2e-5,
        "epochs": 10,
        "patience": 4,
        "dropout": 0.1,
        "model": "bert-base-uncased",
        "n_runs": 3
    }
}
```

Run training:
```bash
cd part_B
python main.py
```

**Note**: This code **only supports training mode**. There is no separate evaluation-only functionality implemented at the moment. The models are automatically evaluated on the test set after training completes.

## Output

Training produces:
- **Models**: Best model saved in `model_bin/experiment_name/`
- **Logs**: Training curves and detailed results
- **Results**: Test set performance with statistical summaries

## Requirements

- PyTorch
- transformers (Hugging Face)
- scikit-learn
- numpy
- pandas
- matplotlib
- tqdm
