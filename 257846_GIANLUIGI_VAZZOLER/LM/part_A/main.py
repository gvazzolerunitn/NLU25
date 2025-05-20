# This file is used to run your functions and print the results
# Please write your fuctions or classes in the functions.py

# Import everything from functions.py file

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

""" EXPERIMENTS = [
    {"name": "LSTM_SGD_dropout_False_lr_1.0_emb_300_hid_200_batch_32", "optimizer": "SGD", "dropout": False, "lr": 1.0, "emb_size": 300, "hid_size": 200, "batch_size": 32},
    {"name": "LSTM_SGD_dropout_True_lr_1.0_emb_300_hid_200_batch_32", "optimizer": "SGD", "dropout": True, "lr": 1.0, "emb_size": 300, "hid_size": 200, "batch_size": 32},
    {"name": "LSTM_SGD_dropout_True_lr_2.0_emb_300_hid_200_batch_32", "optimizer": "SGD", "dropout": True, "lr": 2.0, "emb_size": 300, "hid_size": 200, "batch_size": 32},
    {"name": "LSTM_SGD_dropout_True_lr_2.0_emb_400_hid_300_batch_64", "optimizer": "SGD", "dropout": True, "lr": 2.0, "emb_size": 400, "hid_size": 300, "batch_size": 64},
    {"name": "LSTM_AdamW_dropout_True_lr_0.001_emb_300_hid_200_batch_32", "optimizer": "AdamW", "dropout": True, "lr": 0.001, "emb_size": 300, "hid_size": 200, "batch_size": 32},
    {"name": "LSTM_AdamW_dropout_True_lr_0.0005_emb_300_hid_200_batch_32", "optimizer": "AdamW", "dropout": True, "lr": 0.0005, "emb_size": 300, "hid_size": 200, "batch_size": 32},
    {"name": "LSTM_AdamW_dropout_True_lr_0.0005_emb_400_hid_300_batch_64", "optimizer": "AdamW", "dropout": True, "lr": 0.0005, "emb_size": 400, "hid_size": 300, "batch_size": 64},
    {"name": "LSTM_AdamW_dropout_False_lr_0.001_emb_400_hid_300_batch_64", "optimizer": "AdamW", "dropout": False, "lr": 0.001, "emb_size": 400, "hid_size": 300, "batch_size": 64},
    {"name": "LSTM_SGD_dropout_False_lr_2.0_emb_300_hid_200_batch_64", "optimizer": "SGD", "dropout": False, "lr": 2.0, "emb_size": 300, "hid_size": 200, "batch_size": 64},
    {"name": "LSTM_AdamW_dropout_True_lr_0.001_emb_400_hid_300_batch_32", "optimizer": "AdamW", "dropout": True, "lr": 0.001, "emb_size": 400, "hid_size": 300, "batch_size": 32}
] """

""" common_params = {
    "lr": 0.001,
    "emb_size": 300,
    "hid_size": 200,
    "clip": 5,
    "batch_size": 32,
    "n_epochs": 100,
    "patience": 3
} """




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
    plt.axhline(y=best_ppl, color='r', linestyle='--', label=f'Best Perplexity: {best_ppl:.2f}')
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
        writer = csv.writer(file, delimiter='	')
        writer.writerow(['# Training Configuration'])
        writer.writerow(['Parameter', 'Value'])
        for key, value in config.items():
            writer.writerow([key, value])
        writer.writerow([])
        writer.writerow(['Epoch', 'Train Loss', 'Dev Loss', 'Dev Perplexity'])
        for epoch, loss_tr, loss_dev, ppl in zip(sampled_epochs, losses_train, losses_dev, ppl_devs):
            writer.writerow([epoch, f"{loss_tr:.4f}", f"{loss_dev:.4f}", f"{ppl:.2f}"])
        writer.writerow([])
        writer.writerow(['# Final Evaluation Metrics'])
        writer.writerow(['Final Epoch Validation Perplexity', f"{ppl_devs[-1]:.3f}"])
        writer.writerow(['Best Validation Perplexity', f"{best_ppl:.3f}"])
        writer.writerow(['Test Perplexity', f"{final_ppl:.3f}"])

def save_models(model, best_model, model_dir):
    os.makedirs(model_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(model_dir, 'model.pt'))
    torch.save(best_model.state_dict(), os.path.join(model_dir, 'model_best.pt'))

# ─── your single‐run configuration ───
config = {
    "name":         "TEST",
    "optimizer":    "SGD",        # or "AdamW"
    "emb_size":     300,          # embedding size
    "hid_size":     200,          # hidden size
    "lr":           1.0,          # learning rate
    "batch_size":   32,           # train batch size (eval will match)
    "dropout_rate": 0.4,          # probability for emb & out dropout
    "clip":         5,            # gradient clip
    "n_epochs":     100,          # max epochs
    "patience":     3             # early‐stopping patience
}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using GPU:" if torch.cuda.is_available() else "Using CPU")

    """ for exp in EXPERIMENTS:
        print(f"Running experiment: {exp['name']}")
        model = LM_LSTM(exp['emb_size'], exp['hid_size'], vocab_len,
                        pad_index=lang.word2id['<pad>'],
                        out_dropout=0.2 if exp['dropout'] else 0.0,
                        emb_dropout=0.2 if exp['dropout'] else 0.0).to(device)
        model.apply(init_weights)

        if exp['optimizer'] == "SGD":
            optimizer = torch.optim.SGD(model.parameters(), lr=exp['lr'])
        elif exp['optimizer'] == "AdamW":
            optimizer = torch.optim.AdamW(model.parameters(), lr=exp['lr'])

        criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'])
        criterion_eval = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'], reduction='sum')

        losses_train, losses_dev, ppl_devs, sampled_epochs = [], [], [], []
        best_ppl, patience = math.inf, common_params['patience']
        best_model = None

        for epoch in range(1, common_params['n_epochs'] + 1):
            loss = train_loop(train_loader, optimizer, criterion_train, model, common_params['clip'])
            loss_mean = np.asarray(loss).mean()
            sampled_epochs.append(epoch)
            losses_train.append(loss_mean)
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            losses_dev.append(loss_dev)
            ppl_devs.append(ppl_dev)
            print(f"[Epoch {epoch}] Train loss: {loss_mean:.4f}, Dev loss: {loss_dev:.4f}, Dev ppl: {ppl_dev:.2f}")
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = common_params['patience']
            else:
                patience -= 1
            if patience <= 0:
                break

        best_model.to(device)
        final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
        print(f"Final Test PPL: {final_ppl:.2f}\n")

        run_dir = os.path.join("runs", exp['name'])
        model_dir = os.path.join("model_bin", exp['name'])
        save_models(model, best_model, model_dir)
        plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir)
        save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, exp, ppl_devs, final_ppl) """
    
    exp = config
    print(f"Running experiment: {exp['name']}")

    # 1) get data‐loaders with your chosen batch‐sizes
    train_loader, dev_loader, test_loader = get_loaders(
        batch_size_train=exp["batch_size"],
        batch_size_eval=exp["batch_size"],
        pad_token=lang.word2id["<pad>"],
        device=device
    )

    # 2) build model with your chosen dropout rate
    model = LM_LSTM(
        emb_size    = exp["emb_size"],
        hidden_size = exp["hid_size"],
        output_size = vocab_len,
        pad_index   = lang.word2id["<pad>"],
        emb_dropout = exp["dropout_rate"],
        out_dropout = exp["dropout_rate"]
    ).to(device)
    model.apply(init_weights)

    # 3) pick optimizer
    if exp["optimizer"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), lr=exp["lr"])
    elif exp["optimizer"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])
    else:
        raise ValueError(f"Unknown optimizer: {exp['optimizer']}")

    # 4) loss functions
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval  = torch.nn.CrossEntropyLoss(
        ignore_index=lang.word2id["<pad>"],
        reduction='sum'
    )

    # 5) training loop (unchanged)
    losses_train, losses_dev, ppl_devs, sampled_epochs = [], [], [], []
    best_ppl, patience = math.inf, exp["patience"]
    best_model = None

    for epoch in range(1, exp["n_epochs"] + 1):
        loss = train_loop(train_loader, optimizer, criterion_train, model, exp["clip"])
        sampled_epochs.append(epoch)
        losses_train.append(loss)
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
        losses_dev.append(loss_dev)
        ppl_devs.append(ppl_dev)
        print(f"[Epoch {epoch}] Train loss: {loss:.4f}, Dev loss: {loss_dev:.4f}, Dev ppl: {ppl_dev:.2f}")

        if ppl_dev < best_ppl:
            best_ppl = ppl_dev
            best_model = copy.deepcopy(model).to('cpu')
            patience = exp["patience"]
        else:
            patience -= 1
        if patience <= 0:
            break

    best_model.to(device)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print(f"Final Test PPL: {final_ppl:.2f}\n")

    # 6) save everything
    run_dir   = os.path.join("runs", exp["name"])
    model_dir = os.path.join("model_bin", exp["name"])
    save_models(model, best_model, model_dir)
    plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir)
    save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, exp, ppl_devs, final_ppl)
