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
import sys

# ─── your single‐run configuration ───
config = {
    # mode: "train" to train a fresh model, "eval" to load & test an existing one
    "mode":         "eval",
    # when mode="eval", point to the folder containing model_best.pt; otherwise, set it to 'None'
    "model_dir":    "/home/gian/Documents/AIS/NLU/NLU-exam_assignments/257846_GIANLUIGI_VAZZOLER/LM/part_A/model_bin/TEST"
,

    # the rest drive training when mode="train"
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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    exp = config

    # ─── EVALUATION MODE ───
    if exp["mode"] == "eval":
        if not exp["model_dir"]:
            print("ERROR: Please set config['model_dir'] to your model folder.", file=sys.stderr)
            sys.exit(1)

        # 1) rebuild model
        vocab_len = len(lang.word2id)
        model = LM_LSTM(
            emb_size    = exp["emb_size"],
            hidden_size = exp["hid_size"],
            output_size = vocab_len,
            pad_index   = lang.word2id["<pad>"],
            emb_dropout = exp["dropout_rate"],
            out_dropout = exp["dropout_rate"]
        ).to(device)
        model.apply(init_weights)

        # 2) load saved weights
        ckpt = os.path.join(exp["model_dir"], "model_best.pt")
        model.load_state_dict(torch.load(ckpt, map_location=device))
        print(f"Loaded checkpoint from: {ckpt}\n")

        # 3) prepare test loader
        _, _, test_loader = get_loaders(
            batch_size_train = exp["batch_size"],
            batch_size_eval  = exp["batch_size"],
            pad_token        = lang.word2id["<pad>"],
            device           = DEVICE
        )

        # 4) evaluate
        criterion = torch.nn.CrossEntropyLoss(
            ignore_index = lang.word2id["<pad>"],
            reduction     = "sum"
        )
        test_ppl, test_loss = eval_loop(test_loader, criterion, model)
        print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}")
        sys.exit(0)

    # ─── TRAINING MODE ───
    print(f"Running experiment: {exp['name']}\n")

    # 1) data‐loaders
    train_loader, dev_loader, test_loader = get_loaders(
        batch_size_train = exp["batch_size"],
        batch_size_eval  = exp["batch_size"],
        pad_token        = lang.word2id["<pad>"],
        device           = DEVICE
    )

    # 2) build model
    vocab_len = len(lang.word2id)
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
    else:  # AdamW
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])

    # 4) loss functions
    criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
    criterion_eval  = torch.nn.CrossEntropyLoss(
        ignore_index = lang.word2id["<pad>"],
        reduction     = "sum"
    )

    # 5) training loop with early‐stopping
    losses_train, losses_dev, ppl_devs, sampled_epochs = [], [], [], []
    best_ppl, patience = math.inf, exp["patience"]
    best_model = None

    for epoch in range(1, exp["n_epochs"] + 1):
        loss_train = train_loop(train_loader, optimizer, criterion_train, model, exp["clip"])
        ppl_dev, loss_dev = eval_loop(dev_loader, criterion_eval, model)

        sampled_epochs.append(epoch)
        losses_train.append(loss_train)
        losses_dev.append(loss_dev)
        ppl_devs.append(ppl_dev)

        print(f"[Epoch {epoch}] Train Loss: {loss_train:.4f} | Dev Loss: {loss_dev:.4f} | Dev PPL: {ppl_dev:.2f}")

        if ppl_dev < best_ppl:
            best_ppl   = ppl_dev
            best_model = copy.deepcopy(model).to("cpu")
            patience   = exp["patience"]
        else:
            patience -= 1
            if patience <= 0:
                print("Early stopping.\n")
                break

    # 6) final test evaluation
    best_model.to(device)
    final_ppl, final_loss = eval_loop(test_loader, criterion_eval, best_model)
    print(f"Final Test Loss: {final_loss:.4f} | Test PPL: {final_ppl:.2f}\n")

    # 7) save everything
    run_dir   = os.path.join("runs",      exp["name"])
    model_dir = os.path.join("model_bin", exp["name"])
    save_models(model, best_model, model_dir)
    plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir)
    save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, exp, ppl_devs, final_ppl)


if __name__ == "__main__":
    main()