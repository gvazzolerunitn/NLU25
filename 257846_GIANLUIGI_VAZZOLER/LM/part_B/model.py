# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn
    
# CODE BEFORE VARIATIONAL DROPOUT    
""" class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        assert emb_size == hidden_size
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        lstm_out, _ = self.lstm(emb)

        output = self.output(lstm_out).permute(0,2,1)
        return output """

# DROPOUT CLASS
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        # se non siamo in training o dropout=0, pass-through
        if not self.training or dropout == 0:
            return x
        batch, seq_len, hidden = x.size()
        # 1) maschera diversa per ogni esemplare (batch), fissa lungo seq_len
        mask = x.new_empty(batch, 1, hidden) \
                 .bernoulli_(1 - dropout) \
                 .div_(1 - dropout)
        # 2) espandi su tutta la sequenza temporale
        mask = mask.expand_as(x)
        # 3) applica
        return x * mask

# CODE WITH VARIATIONAL DROPOUT
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, 
                 pad_index=0, 
                 out_dropout=0.1,
                 emb_dropout=0.1, 
                 n_layers=1):
        assert emb_size == hidden_size
        super(LM_LSTM, self).__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        self.input_locked_dropout = LockedDropout()
        self.output_locked_dropout = LockedDropout()

        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)
        self.pad_token = pad_index

        self.output = nn.Linear(hidden_size, output_size)
        self.output.weight = self.embedding.weight  # weight tying

        self.emb_dropout_p = emb_dropout
        self.out_dropout_p = out_dropout

    def forward(self, input_sequence):
        # — Embedding Dropout sui pesi — 
        if self.training and self.emb_dropout_p > 0:
            # 1) crea mask (vocab_size × 1)
            mask = self.embedding.weight.new_empty(
                self.embedding.num_embeddings, 1
                    ).bernoulli_(1 - self.emb_dropout_p) \
            .div_(1 - self.emb_dropout_p)
            # 2) applica la mask sull’intera matrice dei pesi
            emb_weight = self.embedding.weight * mask
        else:
            emb_weight = self.embedding.weight

        # 3) lookup con i pesi “dropout-ati”
        emb = nn.functional.embedding(input_sequence, emb_weight,
                              padding_idx=self.pad_token)

        emb = self.input_locked_dropout(emb, dropout=self.emb_dropout_p)

        lstm_out, _ = self.lstm(emb)
        lstm_out = self.output_locked_dropout(lstm_out, dropout=self.out_dropout_p)

        output = self.output(lstm_out).permute(0, 2, 1)
        return output
