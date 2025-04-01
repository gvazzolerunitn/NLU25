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

def plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_name):
    plt.figure(figsize=(12, 5))

    # Training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(sampled_epochs, losses_train, label='Training Loss')
    plt.scatter(sampled_epochs, losses_train, s=10)
    plt.plot(sampled_epochs, losses_dev, label='Validation Loss')
    plt.scatter(sampled_epochs, losses_dev, s=10)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # Validation perplexity
    plt.subplot(1, 2, 2)
    plt.plot(sampled_epochs, ppl_devs, label='Validation Perplexity', color='orange')
    plt.scatter(sampled_epochs, ppl_devs, s=10, color='orange')
    plt.axhline(y=best_ppl, color='r', linestyle='--', label=f'Best Perplexity: {best_ppl:.2f}')
    plt.xlabel('Epochs')
    plt.ylabel('Perplexity')
    plt.title('Validation Perplexity')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    run_dir = os.path.join('runs', run_name)
    os.makedirs(run_dir, exist_ok=True)

    plot_path = os.path.join(run_dir, 'training_curves.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()

def save_training_log(sampled_epochs, losses_train, losses_dev, run_name, best_ppl, config, ppl_devs, final_ppl):
    run_dir = os.path.join('runs', run_name)
    os.makedirs(run_dir, exist_ok=True)
    csv_path = os.path.join(run_dir, 'training_log.csv')

    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file, delimiter='\t')

        # Config section
        writer.writerow(['# Training Configuration'])
        writer.writerow(['Parameter', 'Value'])
        for key, value in config.items():
            writer.writerow([key, value])
        writer.writerow([])

        # per epoch results section
        writer.writerow(['# Training Results Per Epoch'])
        writer.writerow(['Epoch', 'Training Loss', 'Validation Loss', 'Validation Perplexity'])
        for epoch, loss_tr, loss_dev, ppl in zip(sampled_epochs, losses_train, losses_dev, ppl_devs):
            writer.writerow([
                epoch,
                f"{loss_tr:.6f}",
                f"{loss_dev:.6f}",
                f"{ppl:.3f}"
            ])
        
        # final results section
        writer.writerow([])
        writer.writerow(['# Final Evaluation Metrics'])
        writer.writerow(['Final Epoch Validation Perplexity', f"{ppl_devs[-1]:.3f}"])
        writer.writerow(['Best Validation Perplexity', f"{best_ppl:.3f}"])
        writer.writerow(['Test Perplexity', f"{final_ppl:.3f}"])

if __name__ == "__main__":
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Stai usando la GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Stai usando la CPU")

    n_epochs = 100
    patience = 3
    losses_train = []
    losses_dev = []
    ppl_devs = []
    sampled_epochs = []
    best_ppl = math.inf
    best_model = None
    pbar = tqdm(range(1, n_epochs))

    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_train, model, clip)
        if epoch % 1 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.asarray(loss).mean())
            ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)
            loss_dev_mean = np.asarray(loss_dev).mean()
            losses_dev.append(loss_dev_mean)
            ppl_devs.append(ppl_dev)
            pbar.set_description("PPL: %f" % ppl_dev)
            if ppl_dev < best_ppl:
                best_ppl = ppl_dev
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break

    best_model.to(device)
    final_ppl, _ = eval_loop(test_loader, criterion_eval, best_model)
    print('Test ppl: ', final_ppl)

    # saving models
    path = 'model_bin/model_AdamW_dropout_lr_0.001.pt'
    path_best = 'model_bin/model_AdamW_dropout_lr_0.001_best.pt'
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    torch.save(best_model.state_dict(), path_best)

    # run and parameters config
    run_name = "LSTM_AdamW_dropout_lr_0.001"
    config = {
        "model": model.__class__.__name__,
        "optimizer": optimizer.__class__.__name__,
        "learning_rate": lr,
        "embedding_size": emb_size,
        "hidden_size": hid_size,
        "clip": clip,
        "dropout_embedding": getattr(model, 'emb_dropout', 'N/A'),
        "dropout_output": getattr(model, 'out_dropout', 'N/A'),
        "patience": 3,
        "epochs_ran": len(sampled_epochs),
        "best_ppl": best_ppl
    }

    # plotting and csv saving
    plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_name)
    save_training_log(sampled_epochs, losses_train, losses_dev, run_name, best_ppl, config, ppl_devs, final_ppl)
