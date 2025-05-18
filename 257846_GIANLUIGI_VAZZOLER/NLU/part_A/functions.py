# Add the remaining functions for the model

import os
import copy
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm

from conll import evaluate
from model import *
from utils import *


# Function taken from notebook: weight initialization for RNNs and Linear layers
# This function is designed to improve convergence and stability during training

def init_weights(mat):
    for m in mat.modules():
        if isinstance(m, (nn.GRU, nn.LSTM, nn.RNN)):
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.xavier_uniform_(param[idx * mul:(idx + 1) * mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0] // 4
                        torch.nn.init.orthogonal_(param[idx * mul:(idx + 1) * mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        elif isinstance(m, nn.Linear):
            torch.nn.init.uniform_(m.weight, -0.01, 0.01)
            if m.bias is not None:
                m.bias.data.fill_(0.01)


# Function taken from notebook: single training loop for a batch of data
# This is where the loss for slots and intents is computed and gradients updated

def train_loop(data, optimizer, criterion_slots, criterion_intents, model, clip=5):
    model.train()
    loss_array = []
    for sample in data:
        optimizer.zero_grad()
        slots, intent = model(sample['utterances'], sample['slots_len'])
        loss_intent = criterion_intents(intent, sample['intents'])
        loss_slot = criterion_slots(slots, sample['y_slots'])
        loss = loss_intent + loss_slot
        loss_array.append(loss.item())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
    return loss_array


# Function taken from notebook: evaluation loop with conll F1 and intent classification
# This loop performs predictions and compares them to ground truth for evaluation

def eval_loop(data, criterion_slots, criterion_intents, model, lang):
    model.eval()
    loss_array = []
    ref_intents = []
    hyp_intents = []
    ref_slots = []
    hyp_slots = []

    with torch.no_grad():
        for sample in data:
            slots, intents = model(sample['utterances'], sample['slots_len'])
            loss_intent = criterion_intents(intents, sample['intents'])
            loss_slot = criterion_slots(slots, sample['y_slots'])
            loss = loss_intent + loss_slot
            loss_array.append(loss.item())

            out_intents = [lang.id2intent[x] for x in torch.argmax(intents, dim=1).tolist()]
            gt_intents = [lang.id2intent[x] for x in sample['intents'].tolist()]
            ref_intents.extend(gt_intents)
            hyp_intents.extend(out_intents)

            output_slots = torch.argmax(slots, dim=1)
            for id_seq, seq in enumerate(output_slots):
                length = sample['slots_len'].tolist()[id_seq]
                utt_ids = sample['utterance'][id_seq][:length].tolist()
                gt_ids = sample['y_slots'][id_seq].tolist()
                gt_slots = [lang.id2slot[elem] for elem in gt_ids[:length]]
                utterance = [lang.id2word[elem] for elem in utt_ids]
                to_decode = seq[:length].tolist()
                ref_slots.append([(utterance[id_el], elem) for id_el, elem in enumerate(gt_slots)])
                hyp_slots.append([(utterance[id_el], lang.id2slot[elem]) for id_el, elem in enumerate(to_decode)])

    try:
        results = evaluate(ref_slots, hyp_slots)
    except Exception as ex:
        print("Warning:", ex)
        ref_s = set([x[1] for x in ref_slots])
        hyp_s = set([x[1] for x in hyp_slots])
        print(hyp_s.difference(ref_s))
        results = {"total": {"f": 0}}

    report_intent = classification_report(ref_intents, hyp_intents, zero_division=False, output_dict=True)
    return results, report_intent, loss_array


# CORRECTED: Fixed train function to ensure sampled_epochs is returned
def train(file_path, lang, model, PAD_TOKEN, train_loader, val_loader, test_loader, lr, clip, epochs=200, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()
    
    best_f1 = 0
    best_model = None
    losses_train = []
    losses_val = []
    sampled_epochs = []
    intent_accs = []
    slot_f1s = []
    
    pbar = tqdm(range(1, epochs))
    for epoch in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        if epoch % 5 == 0:
            sampled_epochs.append(epoch)
            losses_train.append(np.mean(loss))
            
            # Evaluation on validation set
            results_val, intent_res, loss_val = eval_loop(val_loader, criterion_slots, criterion_intents, model, lang)
            losses_val.append(np.mean(loss_val))
            f1 = results_val['total']['f']
            intent_acc = intent_res['accuracy']
            
            # Store metrics
            intent_accs.append(intent_acc)
            slot_f1s.append(f1)
            
            # Update best model
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            
            # Print progress
            pbar.set_postfix({
                "Epoch": epoch,
                "Train Loss": np.mean(loss),
                "Val Loss": np.mean(loss_val),
                "Intent Acc": intent_acc,
                "Slot F1": f1
            })
            
            # Early stopping
            if patience <= 0:
                break
    
    # Save the best model
    best_model.to(device)
    to_save = {
        "model": best_model.state_dict(),
        "lang": lang,
    }
    torch.save(to_save, file_path)
    
    # Final evaluation on test set
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
    
    # CORRECTED: Now returning sampled_epochs
    return results_test, intent_test, losses_train, losses_val, intent_accs, slot_f1s, sampled_epochs


# Custom function to save results
def save_run_results(runpath, cfg, sampled_epochs, losses_train, losses_val, intent_accs, slot_f1s, results_test, intent_test):
    # Create directory for this run
    os.makedirs(runpath, exist_ok=True)
    
    # Create a comprehensive CSV file that includes all information
    with open(os.path.join(runpath, 'training_log.csv'), 'w') as f:
        # Write the header section
        f.write("# Training Configuration\n")
        f.write("Parameter\tValue\n")
        
        # Write experiment name based on directory
        experiment_name = os.path.basename(runpath)
        f.write(f"name\t{experiment_name}\n")
        
        # Write all configuration parameters
        for key, value in cfg.items():
            if key != 'run' and key != 'n_runs':  # Skip non-relevant parameters
                f.write(f"{key}\t{value}\n")
        
        # Add empty line before metrics
        f.write("\n")
        
        # Write training metrics header
        f.write("Epoch\tTrain Loss\tVal Loss\tIntent Accuracy\tSlot F1\n")
        
        # Write metrics for each epoch
        for i in range(len(sampled_epochs)):
            f.write(f"{sampled_epochs[i]}\t{losses_train[i]:.4f}\t{losses_val[i]:.4f}\t{intent_accs[i]:.4f}\t{slot_f1s[i]:.4f}\n")
        
        # Add final evaluation metrics
        f.write("\n# Final Evaluation Metrics\n")
        f.write(f"Final Intent Accuracy\t{intent_test['accuracy']:.4f}\n")
        f.write(f"Final Slot F1\t{results_test['total']['f']:.4f}\n")
        
        # Add optional detailed metrics if available
        if 'weighted avg' in intent_test:
            f.write(f"Intent Precision\t{intent_test['weighted avg']['precision']:.4f}\n")
            f.write(f"Intent Recall\t{intent_test['weighted avg']['recall']:.4f}\n")
            f.write(f"Intent F1\t{intent_test['weighted avg']['f1-score']:.4f}\n")
    
    # Still generate the plots for visualization
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(sampled_epochs, losses_train, label='Train Loss')
    plt.plot(sampled_epochs, losses_val, label='Validation Loss')
    plt.title('Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xlim(0, 200)  # Fixed x-axis from 0 to 200
    plt.ylim(0, 3.5)  # Fixed y-axis from 0 to 3.5 for losses
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(sampled_epochs, intent_accs, label='Intent Accuracy')
    plt.plot(sampled_epochs, slot_f1s, label='Slot F1')
    plt.title('Metrics over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.xlim(0, 200)  # Fixed x-axis from 0 to 200
    plt.ylim(0, 1.0)  # Fixed y-axis from 0 to 1.0 for accuracy/F1
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(runpath, 'training_curves.png'))
    plt.close()

# Specific function to save results from multiple runs
def save_aggregated_summary(runpath, experiment_name, slot_f1s, intent_accs):
    filepath = os.path.join(runpath, 'summary.txt')
    with open(filepath, 'w') as f:
        f.write(f"Experiment: {experiment_name}\n")
        f.write(f"Mean Slot F1: {np.mean(slot_f1s):.4f} ± {np.std(slot_f1s):.4f}\n")
        f.write(f"Mean Intent Accuracy: {np.mean(intent_accs):.4f} ± {np.std(intent_accs):.4f}\n")


# CORRECTED: Modified run_training_pipeline to handle the modified train function return value
def run_training_pipeline(experiments_config):
    save_path = "./"
    tmp_train_raw = load_data(os.path.join('dataset', 'ATIS', 'train.json'))
    test_raw = load_data(os.path.join('dataset', 'ATIS', 'test.json'))
    train_raw, val_raw, y_train, y_val, y_test = create_dev_set(tmp_train_raw, test_raw)
    words = sum([x['utterance'].split() for x in train_raw], [])
    corpus = train_raw + val_raw + test_raw
    slots = set(sum([line['slots'].split() for line in corpus], []))
    intents = set([line['intent'] for line in corpus])
    default_options = {
        'hid_size': 200,
        'emb_size': 300,
        'lr': 0.0001,
        'clip': 5,
        'dropout': 0,
        'bidirectional': False,
        'n_runs': 1,
        'run': False,
    }
    
    for experiment in experiments_config:
        cfg = default_options | experiments_config[experiment]
        print(f"Running experiment {experiment}")
        
        if cfg['run']:
            lang = Lang(words, intents, slots, cutoff=0)
        else:
            saved_model = torch.load('./model_bin/' + experiment + '.pt', map_location=torch.device(device))
            lang = saved_model['lang']
        
        out_slot = len(lang.slot2id)
        out_int = len(lang.intent2id)
        vocab_len = len(lang.word2id)
        
        train_dataset = IntentsAndSlots(train_raw, lang)
        val_dataset = IntentsAndSlots(val_raw, lang)
        test_dataset = IntentsAndSlots(test_raw, lang)
        
        train_loader = DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=64, collate_fn=collate_fn)
        
        runpath = os.path.join(save_path, 'runs', experiment)
        os.makedirs(runpath, exist_ok=True)
        os.makedirs('./model_bin', exist_ok=True)
        file_path = './model_bin/' + experiment + '.pt'
        
        slot_f1s, intent_acc = [], []
        results_test, intent_test = [], []
        best_f1_overall = -1  # tracks best overall F1 for best model
        
        if not cfg['run']:
            cfg['n_runs'] = 5
        
        for run_idx in range(cfg['n_runs']):
            model = ModelIAS(cfg['emb_size'], out_slot, out_int, cfg['hid_size'], vocab_len, 
                             pad_index=PAD_TOKEN, bidirectional=cfg['bidirectional'], dropout=cfg['dropout']).to(device)
            
            if cfg['run']:
                model.apply(init_weights)
                # CORRECTED: Updated to accept the new return value from train() including sampled_epochs
                results_test, intent_test, losses_train, losses_val, intent_accs, slot_f1s, sampled_epochs = train(
                    file_path, lang, model, PAD_TOKEN, train_loader, val_loader, test_loader, cfg['lr'], cfg['clip']
                )
                
                # Create a subdirectory for each run
                run_dir = os.path.join(runpath, f"run_{run_idx + 1}")
                os.makedirs(run_dir, exist_ok=True)
                save_run_results(run_dir, cfg, sampled_epochs, losses_train, losses_val, intent_accs, slot_f1s, results_test, intent_test)

                # Save model of this run with unique name
                run_model_path = f'./model_bin/{experiment}_run{run_idx + 1}.pt'
                torch.save({
                    "model": model.state_dict(),
                    "lang": lang,
                }, run_model_path)

                # Save best model among all runs
                if results_test['total']['f'] > best_f1_overall:
                    best_f1_overall = results_test['total']['f']
                    best_model_path = f'./model_bin/{experiment}_best.pt'
                    torch.save({
                        "model": model.state_dict(),
                        "lang": lang,
                    }, best_model_path)

            else:
                model.load_state_dict(saved_model['model'])
                criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                criterion_intents = nn.CrossEntropyLoss()
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)
            
            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])
        
        if cfg['n_runs'] > 1:
            slot_f1s = np.asarray(slot_f1s)
            intent_acc = np.asarray(intent_acc)
            print(f'Slot F1: {slot_f1s.mean():.3f} +- {slot_f1s.std():.3f}')
            print(f'Intent Acc: {intent_acc.mean():.3f} +- {intent_acc.std():.3f}')
            save_aggregated_summary(runpath, experiment, slot_f1s, intent_acc)
        else:
            print(f"Slot F1: {results_test['total']['f']:.3f}")
            print(f"Intent Accuracy: {intent_test['accuracy']:.3f}")