# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

from functions import *

# First experiment: 1 run per configuration

""" experiments_config = {
    "LSTM_dropout0.0_unidir": {
        "dropout": 0.0,
        "bidirectional": False,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.0_bidir": {
        "dropout": 0.0,
        "bidirectional": True,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.1_unidir": {
        "dropout": 0.1,
        "bidirectional": False,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.1_bidir": {
        "dropout": 0.1,
        "bidirectional": True,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.3_unidir": {
        "dropout": 0.3,
        "bidirectional": False,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.3_bidir": {
        "dropout": 0.3,
        "bidirectional": True,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.5_unidir": {
        "dropout": 0.5,
        "bidirectional": False,
        "run": True,
        "n_runs": 1
    },
    "LSTM_dropout0.5_bidir": {
        "dropout": 0.5,
        "bidirectional": True,
        "run": True,
        "n_runs": 1
    }
} """

# Second experiment: 3 runs (with mean) for the two best-performing configurations

experiments_config = {
    "LSTM_dropout0.1_bidir": {
        "dropout": 0.1,
        "bidirectional": True,
        "run": True,
        "n_runs": 3
    },
    "LSTM_dropout0.3_bidir": {
        "dropout": 0.3,
        "bidirectional": True,
        "run": True,
        "n_runs": 3
    }
}

if __name__ == "__main__":
    run_training_pipeline(experiments_config)