# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

from functions import *

if __name__ == "__main__":
    # define experiments
    to_run = {
        "BertLarge_Improved": {
            "run": True,
            "batch_size": 8,        # Smaller batch size for better generalization
            "lr": 1e-5,             # Lower learning rate (half of current)
            "epochs": 10,           # More epochs
            "patience": 4,          # More patience for early stopping
            "clip": 1,              # Lower gradient clipping
            "dropout": 0.3,         # Higher dropout for regularization
            "model": "bert-large-uncased",
            "n_runs": 3
        }
    }
    run_experiments(to_run)
