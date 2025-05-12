# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

""" **Mandatory requirements**: For the following experiments the perplexity must be below 250 (***PPL < 250***) and it should be lower than the one achieved in Part 1.1 (i.e. base LSTM).

Starting from the `LM_RNN` in which you replaced the RNN with a LSTM model, apply the following regularisation techniques:
- Weight Tying 
- Variational Dropout (no DropConnect)
- Non-monotonically Triggered AvSGD 

These techniques are described in [this paper](https://openreview.net/pdf?id=SyyGPP0TZ). """

import os
import math
import copy
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from functions import *
from utils import *
from model import *
import csv

# Experiment configurations
EXPERIMENTS = [
    # STEP 1: Weight Tying only
    {"name": "LSTM_WT_only", "use_vdropout": False, "use_ntasgd": False, "lr": 1.0, "emb_size": 300, "hid_size": 300, "batch_size": 32, "dropout_rate": 0.0},

    # STEP 2: Weight Tying + Variational Dropout
    {"name": "LSTM_WT_VD", "use_vdropout": True, "use_ntasgd": False, "lr": 1.0, "emb_size": 300, "hid_size": 300, "batch_size": 32, "dropout_rate": 0.2},

    # STEP 3: Weight Tying + Variational Dropout + NT-AvSGD (standard config)
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.0, "emb_size": 300, "hid_size": 300, "batch_size": 32, "dropout_rate": 0.2},

    # Exploring lr variations
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.5, "emb_size": 300, "hid_size": 300, "batch_size": 32, "dropout_rate": 0.2},
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 2.0, "emb_size": 300, "hid_size": 300, "batch_size": 32, "dropout_rate": 0.2},

    # Exploring bigger model
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.0, "emb_size": 400, "hid_size": 400, "batch_size": 64, "dropout_rate": 0.2},
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.5, "emb_size": 400, "hid_size": 400, "batch_size": 64, "dropout_rate": 0.2},
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 2.0, "emb_size": 400, "hid_size": 400, "batch_size": 64, "dropout_rate": 0.3},

    # Exploring smaller model
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.0, "emb_size": 200, "hid_size": 200, "batch_size": 32, "dropout_rate": 0.1},
    {"name": "LSTM_WT_VD_NTAvSGD", "use_vdropout": True, "use_ntasgd": True, "lr": 1.5, "emb_size": 200, "hid_size": 200, "batch_size": 32, "dropout_rate": 0.1},
]


# Common parameters
common_params = {
    "clip": 5,
    "n_epochs": 100,
    "patience": 5,  # patience for early stopping when not using NT-AvSGD
    "ntasgd_trigger": 5,  # number of epochs without improvement to start averaging
}

# Utility functions
def plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(sampled_epochs, losses_train, label='Training Loss', marker='o')
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss', marker='o')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.ylim(0, 7)
    plt.xticks(range(0, 101, 10))
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(sampled_epochs, ppl_devs, label='Validation Perplexity', color='orange', marker='o')
    plt.axhline(y=best_ppl, color='r', linestyle='--', label=f'Best PPL: {best_ppl:.2f}')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.ylim(0, 400)
    plt.xticks(range(0, 101, 10))
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    os.makedirs(run_dir, exist_ok=True)
    plt.savefig(os.path.join(run_dir, 'training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, config, ppl_devs, final_ppl):
    csv_path = os.path.join(run_dir, 'training_log.csv')
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerow(['# Training Configuration'])
        writer.writerow(['Parameter', 'Value'])
        for key, value in config.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow(['Epoch', 'Train Loss', 'Dev Loss', 'Dev PPL'])
        for epoch, loss_tr, loss_dev, ppl in zip(sampled_epochs, losses_train, losses_dev, ppl_devs):
            writer.writerow([epoch, f"{loss_tr:.4f}", f"{loss_dev:.4f}", f"{ppl:.2f}"])
        writer.writerow([])
        writer.writerow(['# Final Evaluation Metrics'])
        writer.writerow(['Final Epoch Validation PPL', f"{ppl_devs[-1]:.3f}"])
        writer.writerow(['Best Validation PPL', f"{best_ppl:.3f}"])
        writer.writerow(['Test PPL', f"{final_ppl:.3f}"])

def save_models(model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))

# MAIN
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using GPU:" if torch.cuda.is_available() else "Using CPU")

    for exp in EXPERIMENTS:
        print(f"\nRunning experiment: {exp}")

        # Build model
        model = LM_LSTM(exp['emb_size'], exp['hid_size'], vocab_len,
                        pad_index=lang.word2id['<pad>'],
                        out_dropout=exp['dropout_rate'] if exp['use_vdropout'] else 0.0,
                        emb_dropout=exp['dropout_rate'] if exp['use_vdropout'] else 0.0).to(device)
        model.apply(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=exp['lr'])

        criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'])
        criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'], reduction='sum')

        losses_train, losses_dev, ppl_devs, sampled_epochs = [], [], [], []
        best_ppl = math.inf
        logs = []
        patience = common_params['patience']

        averaging = False
        average_model = None
        trigger_epoch = None

        for epoch in range(1, common_params['n_epochs'] + 1):
            loss = train_loop(train_loader, optimizer, criterion_train, model, common_params['clip'])
            loss_mean = np.asarray(loss).mean()

            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

            sampled_epochs.append(epoch)
            losses_train.append(loss_mean)
            losses_dev.append(loss_dev)
            ppl_devs.append(ppl_dev)

            print(f"[Epoch {epoch}] Train Loss: {loss_mean:.4f}, Dev Loss: {loss_dev:.4f}, Dev PPL: {ppl_dev:.2f}")

            logs.append(loss_dev)

            if not exp['use_ntasgd']:
                # Early stopping
                if ppl_dev < best_ppl:
                    best_ppl = ppl_dev
                    best_model = copy.deepcopy(model)
                    patience = common_params['patience']
                else:
                    patience -= 1
                if patience <= 0:
                    break
            else:
                # NT-AvSGD logic
                if not averaging:
                    if len(logs) > common_params['ntasgd_trigger']:
                        if logs[-1] > min(logs[-common_params['ntasgd_trigger']:]):
                            print(f"Starting Averaging from epoch {epoch}")
                            averaging = True
                            trigger_epoch = epoch
                            average_model = copy.deepcopy(model)
                else:
                    steps_since_trigger = epoch - trigger_epoch
                    for p_avg, p_model in zip(average_model.parameters(), model.parameters()):
                        p_avg.data.mul_(steps_since_trigger / (steps_since_trigger + 1))
                        p_avg.data.add_(p_model.data / (steps_since_trigger + 1))

                if ppl_dev < best_ppl:
                    best_ppl = ppl_dev

        # Final evaluation
        if exp['use_ntasgd'] and averaging:
            average_model.to(device)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, average_model)
            model_to_save = average_model
        else:
            model_to_save = best_model
            model_to_save.to(device)
            final_ppl, _ = eval_loop(test_loader, criterion_eval, model_to_save)

        print(f"Final Test PPL: {final_ppl:.2f}\n")

        # Build the run directory name
        exp_name = f"{exp['name']}_lr{exp['lr']}_emb{exp['emb_size']}_hid{exp['hid_size']}_batch{exp['batch_size']}_dropout{exp['dropout_rate']}"
        run_dir = os.path.join("runs", exp_name)
        model_dir = os.path.join("model_bin", exp_name)

        save_models(model_to_save, model_dir)
        plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir)
        save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, exp, ppl_devs, final_ppl)
