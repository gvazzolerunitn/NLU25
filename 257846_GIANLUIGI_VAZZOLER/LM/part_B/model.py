# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn

# DROPOUT CLASS (needed to implement variational dropout)
# From now the dropout mask is shared across the time steps of a sequence.
class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout, self).__init__()

    def forward(self, x, dropout=0.5):
        # if we are not in training mode or dropout=0, pass-through
        if not self.training or dropout == 0:
            return x
        batch, seq_len, hidden = x.size()
        # 1) different mask for each sample (batch), fixed along seq_len
        mask = x.new_empty(batch, 1, hidden) \
             .bernoulli_(1 - dropout) \
             .div_(1 - dropout)
        # 2) expand across the entire time sequence
        mask = mask.expand_as(x)
        # 3) apply
        return x * mask

# LSTM model with variational dropout
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
        # — Embedding Dropout on weights — 
        if self.training and self.emb_dropout_p > 0:
            # 1) create mask (vocab_size × 1)
            mask = self.embedding.weight.new_empty(
                self.embedding.num_embeddings, 1
                    ).bernoulli_(1 - self.emb_dropout_p) \
            .div_(1 - self.emb_dropout_p)
            # 2) apply the mask to the entire weight matrix
            emb_weight = self.embedding.weight * mask
        else:
            emb_weight = self.embedding.weight

        # 3) lookup with the “dropout-ed” weights
        emb = nn.functional.embedding(input_sequence, emb_weight,
                              padding_idx=self.pad_token)

        emb = self.input_locked_dropout(emb, dropout=self.emb_dropout_p)

        lstm_out, _ = self.lstm(emb)
        lstm_out = self.output_locked_dropout(lstm_out, dropout=self.out_dropout_p)

        output = self.output(lstm_out).permute(0, 2, 1)
        return output
