# This file is used to run your functions and print the results
# Please write your functions or classes in functions.py

import os
import math
import copy
import torch
import numpy as np
from functions import *
import argparse
import sys

# ───────────────────────────────────────────────────────────────────────
# ARGPARSE CONFIGURATION
# ───────────────────────────────────────────────────────────────────────
def get_config():
    p = argparse.ArgumentParser(
        description="Part B training or evaluation of your LM_LSTM experiments"
    )
    p.add_argument(
        "--mode",
        choices=["train","eval"],
        default="train",
        help="Mode: 'train' to run all EXPERIMENTS, 'eval' to load & test one."
    )
    p.add_argument(
        "--eval_exp",
        type=str,
        default=None,
        help="Name of one EXPERIMENT to evaluate (required if --mode=eval)."
    )
    p.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to folder with 'model.pt' for that experiment."
    )
    return vars(p.parse_args())

# ───────────────────────────────────────────────────────────────────────
# Experiment configurations
# ───────────────────────────────────────────────────────────────────────
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
    "clip":           5,
    "n_epochs":      100,
    "patience":       5,   # for standard early‐stop
    "ntasgd_trigger": 5,   # epochs without improvement before averaging
}

def make_run_name(exp):
    """Helper to generate consistent run names"""
    return (
        f"{exp['name']}"
        f"_lr{exp['lr']}"
        f"_emb{exp['emb_size']}"
        f"_hid{exp['hid_size']}"
        f"_batch{exp['batch_size']}"
        f"_dropout{exp['dropout_rate']}"
    )

def run_evaluation(config):
    """Handle evaluation mode"""
    provided_dir = config["model_dir"]
    if not provided_dir:
        print("ERROR: --mode eval requires --model_dir", file=sys.stderr)
        sys.exit(1)

    run_name = os.path.basename(provided_dir)
    print(f"\n[EVAL MODE] Loading run '{run_name}' from {provided_dir}\n")

    # find the matching experiment
    matched = None
    for exp in EXPERIMENTS:
        if make_run_name(exp) == run_name:
            matched = exp
            break

    if matched is None:
        print(f"ERROR: no experiment in EXPERIMENTS generated run_name '{run_name}'", file=sys.stderr)
        sys.exit(1)

    exp = matched
    print("Matched hyper-params:", exp, "\n")

    # 1) rebuild the exact model
    vocab_len = len(lang.word2id)
    model = LM_LSTM(
        exp["emb_size"], exp["hid_size"], vocab_len,
        pad_index   = lang.word2id["<pad>"],
        emb_dropout = exp["dropout_rate"] if exp["use_vdropout"] else 0.0,
        out_dropout = exp["dropout_rate"] if exp["use_vdropout"] else 0.0
    ).to(DEVICE)
    model.apply(init_weights)

    # 2) load weights
    ckpt = os.path.join(provided_dir, "model.pt")
    model.load_state_dict(torch.load(ckpt, map_location=DEVICE))
    print(f"✔ Loaded checkpoint: {ckpt}\n")

    # 3) fresh test_loader with correct batch‐size
    _, _, test_loader = get_loaders(
        batch_size_train = exp["batch_size"],
        batch_size_eval  = exp["batch_size"],
        pad_token        = lang.word2id["<pad>"],
        device           = DEVICE
    )

    # 4) eval
    criterion = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction="sum")
    test_ppl, test_loss = eval_loop(test_loader, criterion, model)
    print(f"Test Loss: {test_loss:.4f} | Test PPL: {test_ppl:.2f}\n")

def run_training():
    """Handle training mode"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")

    for exp in EXPERIMENTS:
        print(f"\nRunning experiment: {exp['name']}")

        # Build model
        model = LM_LSTM(
            exp['emb_size'], exp['hid_size'], len(lang.word2id),
            pad_index   = lang.word2id['<pad>'],
            emb_dropout = exp['dropout_rate'] if exp['use_vdropout'] else 0.0,
            out_dropout = exp['dropout_rate'] if exp['use_vdropout'] else 0.0
        ).to(device)
        model.apply(init_weights)

        optimizer = torch.optim.SGD(model.parameters(), lr=exp['lr'])

        criterion_train = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'])
        criterion_eval  = torch.nn.CrossEntropyLoss(ignore_index=lang.word2id['<pad>'], reduction='sum')

        # get loaders
        train_loader, dev_loader, test_loader = get_loaders(
            batch_size_train = exp["batch_size"],
            batch_size_eval  = exp["batch_size"],
            pad_token        = lang.word2id["<pad>"],
            device           = DEVICE
        )

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
                    if len(logs) > common_params['ntasgd_trigger'] and logs[-1] > min(logs[-common_params['ntasgd_trigger']:]):
                        print(f"Starting Averaging from epoch {epoch}")
                        averaging = True
                        trigger_epoch = epoch
                        average_model = copy.deepcopy(model)
                else:
                    steps = epoch - trigger_epoch
                    for p_avg, p_model in zip(average_model.parameters(), model.parameters()):
                        p_avg.data.mul_(steps / (steps + 1))
                        p_avg.data.add_(p_model.data / (steps + 1))

                if ppl_dev < best_ppl:
                    best_ppl = ppl_dev

        # Final evaluation
        if exp['use_ntasgd'] and averaging:
            average_model.to(device)
            final_model = average_model
        else:
            final_model = best_model.to(device)

        final_ppl, _ = eval_loop(test_loader, criterion_eval, final_model)
        print(f"Final Test PPL: {final_ppl:.2f}\n")

        # Build directories & save
        exp_name  = make_run_name(exp)
        run_dir   = os.path.join("runs",      exp_name)
        model_dir = os.path.join("model_bin", exp_name)

        save_models(final_model, model_dir)  # note: model.pt & model_best.pt
        plot_training_curves(sampled_epochs, losses_train, losses_dev, ppl_devs, best_ppl, run_dir)
        save_training_log(sampled_epochs, losses_train, losses_dev, run_dir, best_ppl, exp, ppl_devs, final_ppl)

if __name__ == "__main__":
    config = get_config()
    
    if config["mode"] == "eval":
        run_evaluation(config)
    else:  # train mode
        run_training()
