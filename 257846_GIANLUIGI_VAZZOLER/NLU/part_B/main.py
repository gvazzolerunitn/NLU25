# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

from functions import *

if __name__ == "__main__":
    # define experiments; 'batch_size' added for flexibility
    to_run = {
        #'BertJoint':          {'run': True,  'batch_size': 32, 'lr':5e-5, 'epochs':30, 'patience':5, 'clip':5, 'dropout':0.1},
        #'BertJoint_dropout':  {'run': True,  'batch_size': 32, 'lr':5e-5, 'epochs':30, 'patience':5, 'clip':5, 'dropout':0.5},
        'BertJoint_best':  {'run': True,  'batch_size': 32, 'lr':5e-5, 'epochs':10, 'patience':3, 'clip':5, 'dropout':0.1}
    }
    run_experiments(to_run)
