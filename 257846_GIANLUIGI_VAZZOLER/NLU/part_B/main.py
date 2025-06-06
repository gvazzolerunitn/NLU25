# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

from functions import *

if __name__ == "__main__":
    # define experiments
    to_run = {
        "BertJoint": {
            "run": False,
            "batch_size": 32,        # Use smaller batch size for better generalization
            "lr": 1e-5,             
            "epochs": 10,           # BERT is pretrained, so fewer epochs are needed
            "patience": 4,         
            "clip": 1,              
            "dropout": 0.1,         
            "model": "bert-base-uncased", # or change to bert-large-uncased
            "n_runs": 3
        }
    }
    run_experiments(to_run)
