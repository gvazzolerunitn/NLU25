# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

from functions import *

experiments_config = {
    "LSTM_dropout_unidir": {
        "dropout": 0.3,
        "bidirectional": False,
        "run": True,
        "n_runs": 1
    }
}

if __name__ == "__main__":
    run_training_pipeline(experiments_config)