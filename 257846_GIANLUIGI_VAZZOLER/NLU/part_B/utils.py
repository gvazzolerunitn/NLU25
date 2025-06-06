# Add functions or classes used for data loading and preprocessing

import os
import json
import torch
from collections import Counter
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast

PAD_TOKEN = -100                # ignore_index for slot loss
DEVICE    = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def load_data(path):
    with open(path) as f:
        return json.load(f)

def divide_training_set(tmp_train_raw, test_raw):
    """
    Split 10% of train_raw into a dev set, stratified by intent.
    """
    portion = 0.10
    intents = [x['intent'] for x in tmp_train_raw]
    count_y = Counter(intents)

    inputs, labels, mini_train = [], [], []
    for idx, intent in enumerate(intents):
        if count_y[intent] > 1:
            inputs.append(tmp_train_raw[idx])
            labels.append(intent)
        else:
            mini_train.append(tmp_train_raw[idx])

    X_train, X_val, y_train, y_val = train_test_split(
        inputs, labels,
        test_size=portion,
        random_state=42,
        shuffle=True,
        stratify=labels
    )
    X_train.extend(mini_train)
    y_test = [x['intent'] for x in test_raw]
    return X_train, X_val, y_train, y_val, y_test

# Here choose between 'bert-base-uncased' or 'bert-large-uncased'
def get_tokenizer(model_name='bert-large-uncased'):
    return BertTokenizerFast.from_pretrained(model_name) # returns a huggingface tokenizer

# Custom function to align word-level labels with BERT subword tokens
# Handles BERT's subword tokenization by mapping original word labels to token positions
def align_labels(tokenized_inputs, word_labels, label2id, pad_token_label_id=PAD_TOKEN):
    aligned = []
    for i, labels in enumerate(word_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        prev_w  = None
        lab_ids = []
        for w in word_ids:
            if w is None: # Special tokens ([CLS], [SEP], [PAD])
                lab_ids.append(pad_token_label_id)
            elif w != prev_w: # First subword of a word gets the original label
                lab_ids.append(label2id[labels[w]])
            else: # Subsequent subwords get padding label
                lab_ids.append(pad_token_label_id)
            prev_w = w
        aligned.append(lab_ids)
    return aligned

# Custom Dataset class for joint intent classification and slot filling
# Handles BERT tokenization and label alignment for dual NLU tasks
class NLUDataset(torch.utils.data.Dataset):
    def __init__(self, raw_data, tokenizer, slot2id, intent2id, max_len=50):
        # Extract data components
        self.texts   = [x['utterance'].split() for x in raw_data]
        self.slots   = [x['slots'].split()     for x in raw_data]
        self.intents = [x['intent']             for x in raw_data]
        self.tokenizer  = tokenizer
        self.slot2id    = slot2id
        self.intent2id  = intent2id
        self.max_len    = max_len

        # Custom BERT tokenization for pre-split words
        self.encodings = tokenizer(
            self.texts,                     # e.g. [ ["i","want","to","fly"], ["book","a","flight"] … ]
            is_split_into_words=True,       # crucial for proper alignment
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt'
        )
        # Custom label alignment and conversion
        self.slot_labels   = align_labels(self.encodings, self.slots, self.slot2id)
        self.intent_labels = [self.intent2id[i] for i in self.intents]

    def __len__(self):
        return len(self.intents)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item['labels_slots']  = torch.tensor(self.slot_labels[idx])
        item['labels_intent'] = torch.tensor(self.intent_labels[idx])
        return item
