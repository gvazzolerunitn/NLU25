# Add the remaining functions for the model

# File: functions.py

# Add the remaining functions for the model (from original notebook)

import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime
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


# Adapted from notebook logic: high-level train function with early stopping and save best model
# Combines all training epochs and evaluates on validation set every 5 epochs

def train(file_path, lang, model, PAD_TOKEN, train_loader, val_loader, test_loader, lr, clip, epochs=200, patience=3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
    criterion_intents = nn.CrossEntropyLoss()

    best_f1 = 0
    best_model = None
    losses_train = []
    losses_val = []
    sampled_epochs = []

    pbar = tqdm(range(1, epochs))
    for x in pbar:
        loss = train_loop(train_loader, optimizer, criterion_slots, criterion_intents, model, clip=clip)
        if x % 5 == 0:
            sampled_epochs.append(x)
            losses_train.append(np.mean(loss))
            results_val, intent_res, loss_val = eval_loop(val_loader, criterion_slots, criterion_intents, model, lang)
            losses_val.append(np.mean(loss_val))

            f1 = results_val['total']['f']
            if f1 > best_f1:
                best_f1 = f1
                best_model = copy.deepcopy(model).to('cpu')
                patience = 3
            else:
                patience -= 1
            if patience <= 0:
                break

    best_model.to(device)
    to_save = {
        "model": best_model.state_dict(),
        "lang": lang,
    }
    torch.save(to_save, file_path)
    results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, best_model, lang)
    return results_test, intent_test


# Handles dataset preparation, model instantiation, training and saving results

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
            saved_model = torch.load('./bin/' + experiment + '.pt', map_location=torch.device(device))
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

        d = datetime.now()
        strftime = d.strftime("%Y-%m-%d_%H-%M")
        runpath = save_path + 'runs/' + experiment + '/' + strftime + '/'
        os.makedirs(runpath, exist_ok=True)
        os.makedirs('./bin', exist_ok=True)
        file_path = './bin/' + experiment + '.pt'

        slot_f1s, intent_acc = [], []
        results_test, intent_test = [], []
        if not cfg['run']:
            cfg['n_runs'] = 5

        for _ in range(cfg['n_runs']):
            model = ModelIAS(cfg['emb_size'], out_slot, out_int, cfg['hid_size'], vocab_len, 
                             pad_index=PAD_TOKEN, bidirectional=cfg['bidirectional'], dropout=cfg['dropout']).to(device)
            if cfg['run']:
                model.apply(init_weights)
                results_test, intent_test = train(file_path, lang, model, PAD_TOKEN, train_loader, val_loader, 
                                                  test_loader, cfg['lr'], cfg['clip'])
            else:
                model.load_state_dict(saved_model['model'])
                criterion_slots = nn.CrossEntropyLoss(ignore_index=PAD_TOKEN)
                criterion_intents = nn.CrossEntropyLoss()
                results_test, intent_test, _ = eval_loop(test_loader, criterion_slots, criterion_intents, model, lang)

            intent_acc.append(intent_test['accuracy'])
            slot_f1s.append(results_test['total']['f'])

        if cfg['run']:
            f = open(runpath + 'results.txt', "a")

        if cfg['n_runs'] > 1:
            slot_f1s = np.asarray(slot_f1s)
            intent_acc = np.asarray(intent_acc)

            print(f'Slot F1: {slot_f1s.mean():.3f} +- {slot_f1s.std():.3f}')
            print(f'Intent Acc: {intent_acc.mean():.3f} +- {intent_acc.std():.3f}')
            if cfg['run']:
                f.write(f'Slot F1: {slot_f1s.mean():.3f} +- {slot_f1s.std():.3f}\nIntent Acc: {intent_acc.mean():.3f} '
                        f'+- {intent_acc.std():.3f}\n')
        else:
            if not cfg['run'] and saved_model.get('results'):
                print(f"[AVG] Slot F1: {saved_model['results']['Slot F1']}")
                print(f"[AVG] Intent Accuracy: {saved_model['results']['Intent Acc']}")
            print(f"Slot F1: {results_test['total']['f']:.3f}")
            print(f"Intent Accuracy: {intent_test['accuracy']:.3f}")
            if cfg['run']:
                f.write(f"Slot F1: {results_test['total']['f']:.3f}\nIntent Accuracy: {intent_test['accuracy']:.3f}\n")

        if cfg['run']:
            f.close()
