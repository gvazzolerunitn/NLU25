import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import matplotlib.pyplot as plt
import csv

from tqdm import tqdm

from model import *
from utils import *

def train_loop(data, optimizer, criterion, model, clip=5):
    model.train()
    loss_array = []
    number_of_tokens = []
    
    for sample in data:
        optimizer.zero_grad() # Zeroing the gradient
        output = model(sample['source'])
        loss = criterion(output, sample['target'])
        loss_array.append(loss.item() * sample["number_tokens"])
        number_of_tokens.append(sample["number_tokens"])
        loss.backward() # Compute the gradient, deleting the computational graph
        # clip the gradient to avoid exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  
        optimizer.step() # Update the weights
        
    return sum(loss_array)/sum(number_of_tokens)

def eval_loop(data, eval_criterion, model):
    model.eval()
    loss_to_return = []
    loss_array = []
    number_of_tokens = []
    # softmax = nn.Softmax(dim=1) # Use Softmax if you need the actual probability
    with torch.no_grad(): # It used to avoid the creation of computational graph
        for sample in data:
            output = model(sample['source'])
            loss = eval_criterion(output, sample['target'])
            loss_array.append(loss.item())
            number_of_tokens.append(sample["number_tokens"])
            
    ppl = math.exp(sum(loss_array) / sum(number_of_tokens))
    loss_to_return = sum(loss_array) / sum(number_of_tokens)
    return ppl, loss_to_return

def init_weights(mat):
    for m in mat.modules():
        if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
            for name, param in m.named_parameters():
                if 'weight_ih' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.xavier_uniform_(param[idx*mul:(idx+1)*mul])
                elif 'weight_hh' in name:
                    for idx in range(4):
                        mul = param.shape[0]//4
                        torch.nn.init.orthogonal_(param[idx*mul:(idx+1)*mul])
                elif 'bias' in name:
                    param.data.fill_(0)
        else:
            if type(m) in [nn.Linear]:
                torch.nn.init.uniform_(m.weight, -0.01, 0.01)
                if m.bias != None:
                    m.bias.data.fill_(0.01)

""" # Experiment also with a smaller or bigger model by changing hid and emb sizes 
# A large model tends to overfit
hid_size = 300 # OLD: 200 (they must be the same value for weight tying)
emb_size = 300

# Don't forget to experiment with a lower training batch size
# Increasing the back propagation steps can be seen as a regularization step

# With SGD try with an higher learning rate (> 1 for instance)
lr = 1 # This is definitely not good for SGD [try 1 for SGD and 0.001 for AdamW]
clip = 5 # Clip the gradient

vocab_len = len(lang.word2id)

model = LM_LSTM(emb_size, hid_size, vocab_len, pad_index=lang.word2id["<pad>"]).to(DEVICE)
model.apply(init_weights)

# OLD CODE => SGD
optimizer = optim.SGD(model.parameters(), lr=lr)

criterion_train = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"])
criterion_eval = nn.CrossEntropyLoss(ignore_index=lang.word2id["<pad>"], reduction='sum')

# Dataloader instantiation
# You can reduce the batch_size if the GPU memory is not enough
train_loader = DataLoader(train_dataset, batch_size=32, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]),  shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"]))
test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=partial(collate_fn, pad_token=lang.word2id["<pad>"])) """

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
