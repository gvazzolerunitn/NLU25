# Add the remaining functions for the model

import os, copy
import numpy as np
import torch
import random
import torch.nn as nn
from torch.utils.data import DataLoader
from datetime import datetime
from sklearn.metrics import classification_report
from conll import evaluate

import matplotlib.pyplot as plt
import pandas as pd
from tqdm.auto import tqdm, trange

from model import *
from utils import *

# ─────────── 0) setting seed ───────────
def set_global_seed(seed: int):
    """
    Sets the seed for Python, NumPy, and PyTorch (CPU/GPU),
    and disables non-deterministic behaviors in cuDNN.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ─────────── 1) Training loop with tqdm ───────────
def train_loop(model, loader, optimizer, scheduler,
               crit_slot, crit_intent, device, clip=5):
    model.train()
    total_loss = 0.0
    for batch in tqdm(loader, desc="Training batches", leave=False):
        optimizer.zero_grad()
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)

        intent_logits, slot_logits = model(input_ids, attention_mask)
        loss_intent = crit_intent(
            intent_logits,
            batch['labels_intent'].to(device)
        )
        loss_slot = crit_slot(
            slot_logits.view(-1, slot_logits.size(-1)),
            batch['labels_slots'].view(-1).to(device)
        )
        (loss_intent + loss_slot).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        total_loss += (loss_intent + loss_slot).item()
    return total_loss / len(loader)

# ─────────── 2) Eval loop returning val_loss ───────────
def eval_loop(model, loader, crit_slot, crit_intent,
              id2slot, id2intent, pad_id, device):
    model.eval()
    val_losses = []
    all_int_preds, all_int_labels = [], []
    refs, hyps = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Validation batches", leave=False):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            intent_logits, slot_logits = model(input_ids, attention_mask)
            loss_intent = crit_intent(
                intent_logits,
                batch['labels_intent'].to(device)
            )
            loss_slot = crit_slot(
                slot_logits.view(-1, slot_logits.size(-1)),
                batch['labels_slots'].view(-1).to(device)
            )
            val_losses.append((loss_intent + loss_slot).item())

            # intent preds & labels
            preds_i = intent_logits.argmax(dim=1).cpu().tolist()
            labs_i  = batch['labels_intent'].tolist()
            all_int_preds.extend(preds_i)
            all_int_labels.extend(labs_i)

            # slot preds & labels
            slot_preds = slot_logits.argmax(dim=2).cpu().tolist()
            slot_trues = batch['labels_slots'].cpu().tolist()
            masks      = attention_mask.cpu().tolist()
            for p_seq, t_seq, m_seq in zip(slot_preds, slot_trues, masks):
                r, h = [], []
                for p, t, m in zip(p_seq, t_seq, m_seq):
                    if m == 0:
                        break
                    if t == pad_id:
                        continue
                    r.append(id2slot[t])
                    h.append(id2slot[p])
                refs.append(r)
                hyps.append(h)

    slot_res = evaluate(
        [[('', w) for w in seq] for seq in refs],
        [[('', w) for w in seq] for seq in hyps]
    )
    intent_rep = classification_report(
        [id2intent[i] for i in all_int_labels],
        [id2intent[i] for i in all_int_preds],
        output_dict=True, zero_division=False
    )

    mean_val_loss = float(np.mean(val_losses))
    return slot_res, intent_rep, mean_val_loss

# ─────────── 3) Helper to save CSV + plot ───────────
def save_run_results(runpath, cfg, epochs, train_losses, val_losses,
                     intent_accs, slot_f1s):
    os.makedirs(runpath, exist_ok=True)

    # a) CSV log
    df = pd.DataFrame({
        'epoch':           epochs,
        'train_loss':      train_losses,
        'val_loss':        val_losses,
        'intent_accuracy': intent_accs,
        'slot_f1':         slot_f1s
    })
    csv_path = os.path.join(runpath, 'training_log.csv')
    with open(csv_path, 'w') as f:
        for k, v in cfg.items():
            if k != 'run':
                f.write(f"# {k}={v}\n")
        df.to_csv(f, index=False)

    # b) Summary.txt file
    summary_path = os.path.join(runpath, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write("=== EXPERIMENT SUMMARY ===\n\n")
        
        # Configuration
        f.write("Configuration:\n")
        for k, v in cfg.items():
            if k not in ['run', 'runpath', 'model_path']:
                f.write(f"  {k}: {v}\n")
        f.write("\n")
        
        # Training Summary
        f.write("Training Summary:\n")
        f.write(f"  Total Epochs: {len(epochs)}\n")
        f.write(f"  Final Train Loss: {train_losses[-1]:.4f}\n")
        f.write(f"  Final Val Loss: {val_losses[-1]:.4f}\n")
        f.write(f"  Best Slot F1: {max(slot_f1s):.4f}\n")
        f.write(f"  Best Intent Acc: {max(intent_accs):.4f}\n")
        f.write(f"  Final Slot F1: {slot_f1s[-1]:.4f}\n")
        f.write(f"  Final Intent Acc: {intent_accs[-1]:.4f}\n")

    # c) Plot curves (smoothed, fixed axes)
    epochs_arr    = np.array(epochs)
    train_smooth  = pd.Series(train_losses).rolling(window=3, center=True, min_periods=1).mean()
    val_smooth    = pd.Series(val_losses).rolling(window=3, center=True, min_periods=1).mean()
    intent_smooth = pd.Series(intent_accs).rolling(window=3, center=True, min_periods=1).mean()
    slot_smooth   = pd.Series(slot_f1s).rolling(window=3, center=True, min_periods=1).mean()

    plt.figure(figsize=(12, 5))
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs_arr, train_smooth,  marker='o', label='Train Loss')
    plt.plot(epochs_arr, val_smooth,    marker='o', label='Val Loss')
    plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.xlim(1, cfg.get('epochs', epochs_arr.max()))
    plt.ylim(0, 1.6)
    plt.xticks(epochs_arr)

    # Metrics plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs_arr, intent_smooth, marker='o', label='Intent Acc')
    plt.plot(epochs_arr, slot_smooth,   marker='o', label='Slot F1')
    plt.xlabel('Epoch'); plt.ylabel('Score'); plt.legend()
    plt.xlim(1, cfg.get('epochs', epochs_arr.max()))
    plt.ylim(0, 1.0)
    plt.xticks(epochs_arr)

    plt.tight_layout()
    plt.savefig(os.path.join(runpath, 'training_curves.png'))
    plt.close()

# ─────────── 4) Training + saving pipeline ───────────
def train_and_save(runpath, model, train_loader, val_loader, test_loader,
                   slot2id, intent2id, cfg):
    device = DEVICE
    model  = model.to(device)

    optimizer   = torch.optim.Adam(model.parameters(), lr=cfg['lr'], eps=1e-8)
    crit_intent = nn.CrossEntropyLoss()
    crit_slot   = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)

    best_f1, best_model = 0.0, None
    patience            = cfg['patience']

    epochs       = cfg['epochs']
    epoch_list   = []
    train_losses = []
    val_losses   = []
    intent_accs  = []
    slot_f1s     = []

    # Epoch loop with tqdm
    for ep in trange(1, epochs+1, desc="Epochs", leave=True):
        # Training step
        tloss = train_loop(
            model, train_loader, optimizer, None,
            crit_slot, crit_intent, device,
            cfg.get('clip', 5)
        )

        # Validation step
        slot_val, intent_val, vloss = eval_loop(
            model, val_loader,
            crit_slot, crit_intent,
            {v:k for k,v in slot2id.items()},
            {v:k for k,v in intent2id.items()},
            PAD_TOKEN, device
        )

        # Record metrics
        epoch_list.append(ep)
        train_losses.append(tloss)
        val_losses.append(vloss)
        intent_accs.append(intent_val['accuracy'])
        slot_f1s.append(slot_val['total']['f'])

        # Early stopping
        f1 = slot_val['total']['f']
        if f1 > best_f1:
            best_f1    = f1
            best_model = copy.deepcopy(model).cpu()
            patience   = cfg['patience']
        else:
            patience -= 1
        if patience <= 0:
            break

    # Save CSV + plots
    save_run_results(runpath, cfg,
                     epoch_list, train_losses, val_losses,
                     intent_accs, slot_f1s)

    # Final test evaluation
    model.load_state_dict(best_model.state_dict())
    slot_test, intent_test, _ = eval_loop(
        model, test_loader,
        crit_slot, crit_intent,
        {v:k for k,v in slot2id.items()},
        {v:k for k,v in intent2id.items()},
        PAD_TOKEN, device
    )

    # Append final test metrics to CSV
    csv_path = os.path.join(runpath, 'training_log.csv')
    with open(csv_path, 'a') as f:
        f.write("\n# Final Test Metrics\n")
        f.write(f"test_intent_accuracy,{intent_test['accuracy']:.4f}\n")
        f.write(f"test_slot_f1,{slot_test['total']['f']:.4f}\n")

    # Save best model
    os.makedirs(os.path.dirname(cfg['model_path']), exist_ok=True)
    torch.save({'model': best_model.state_dict()}, cfg['model_path'])

    return slot_test, intent_test

# ─────────── 5) run_experiments ───────────
def run_experiments(to_run):
    tmp_train = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw  = load_data(os.path.join('dataset', 'ATIS', 'test.json'))
    train_raw, val_raw, _, _, _ = divide_training_set(tmp_train, test_raw)

    # Build label maps
    corpus    = train_raw + val_raw + test_raw
    slots     = sorted({s for x in corpus for s in x['slots'].split()})
    intents   = sorted({x['intent'] for x in corpus})
    slot2id   = {l:i for i,l in enumerate(slots)}
    intent2id = {l:i for i,l in enumerate(intents)}

    tokenizer = get_tokenizer()

    # Prepare datasets once
    train_ds = NLUDataset(train_raw, tokenizer, slot2id, intent2id)
    val_ds   = NLUDataset(val_raw,   tokenizer, slot2id, intent2id)
    test_ds  = NLUDataset(test_raw,  tokenizer, slot2id, intent2id)

    base_seed = 42  # starting seed for reproducibility

    for exp, cfg in to_run.items():
        timestamp  = datetime.now().strftime("%Y%m%d_%H%M%S")
        runpath    = os.path.join("runs",    exp, timestamp)
        model_path = os.path.join("model_bin", exp + ".pt")
        cfg.update({
            'runpath':    runpath,
            'model_path': model_path,
            'model_name': 'bert-large-uncased' # or 'bert-large-uncased'
        })
        os.makedirs(runpath, exist_ok=True)

        # By default, if you haven't set 'n_runs' in cfg, use 1
        n_runs = cfg.get('n_runs', 1)

        # Container to collect test metrics across all runs
        test_slot_f1s = []
        test_intent_accs = []

        print(f"\n===== Running {exp} ({n_runs} runs) =====")
        for run_idx in range(n_runs):
            current_seed = base_seed + run_idx
            set_global_seed(current_seed)
            print(f"  Run {run_idx+1}/{n_runs} with seed {current_seed}")

            # Instantiate and train/evaluate the model
            train_loader = DataLoader(train_ds,
                                      batch_size=cfg['batch_size'],
                                      shuffle=True)
            val_loader   = DataLoader(val_ds,
                                      batch_size=cfg['batch_size'])
            test_loader  = DataLoader(test_ds,
                                      batch_size=cfg['batch_size'])

            # Build an instance of JointBertForNLU
            model = JointBertForNLU(
                cfg['model_name'],
                len(intents),
                len(slots),
                cfg['dropout']
            )

            slot_test, intent_test = train_and_save(
                runpath,
                model,
                train_loader, val_loader, test_loader,
                slot2id, intent2id,
                cfg
            )
            print(f"    → Test Slot F1: {slot_test['total']['f']:.4f}")
            print(f"    → Test Intent Acc: {intent_test['accuracy']:.4f}")

            test_slot_f1s.append(slot_test['total']['f'])
            test_intent_accs.append(intent_test['accuracy'])

        # If n_runs > 1, print mean ± std
        if n_runs > 1:
            arr_slot = np.asarray(test_slot_f1s)
            arr_int  = np.asarray(test_intent_accs)
            print(f"\n  [{exp} Summary] Slot F1: {arr_slot.mean():.4f} ± {arr_slot.std():.4f}")
            print(f"  [{exp} Summary] Intent Acc: {arr_int.mean():.4f} ± {arr_int.std():.4f}")

