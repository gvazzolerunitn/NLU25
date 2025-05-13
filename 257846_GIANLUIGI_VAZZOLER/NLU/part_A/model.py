# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class ModelIAS(nn.Module):

    def __init__(self, hid_size, out_slot, out_int, emb_size, vocab_len,
                 pad_index=0, bidirectional=False, dropout=0.1, n_layer=1):
        """
        Modified version of the original ModelIAS to support bidirectional LSTM.

        Changes from original:
        - Added 'bidirectional' and 'dropout' as configurable parameters.
        - Updated LSTM output handling for bidirectional case (double hidden size).
        - Improved final intent representation by concatenating forward/backward final states when bidirectional=True.
        """
        super(ModelIAS, self).__init__()
        # hid_size = Hidden size
        # out_slot = number of slots (output size for slot filling)
        # out_int = number of intents (output size for intent class)
        # emb_size = word embedding size

        # Store configuration
        self.bidirectional = bidirectional
        self.hid_size = hid_size  # Needed to compute encoding size if bidirectional

        # Embedding layer: converts word indices into dense vectors
        self.embedding = nn.Embedding(vocab_len, emb_size, padding_idx=pad_index)

        # LSTM encoder: can be unidirectional or bidirectional based on config
        self.utt_encoder = nn.LSTM(emb_size, hid_size, n_layer,
                                   bidirectional=bidirectional,  # NEW: Configurable directionality
                                   batch_first=True)

        # Calculate effective encoding size based on directionality
        encoding_size = hid_size * 2 if bidirectional else hid_size  # NEW: Double size if bidirectional

        # Output heads
        self.slot_out = nn.Linear(encoding_size, out_slot)  # Slot filling head
        self.intent_out = nn.Linear(encoding_size, out_int)  # Intent classification head

        # Dropout layer used during training (shared)
        self.dropout = nn.Dropout(dropout)


    def forward(self, utterance, seq_lengths):
        """
        Forward pass through the network.

        In this modified version:
        - Bidirectional flag affects how the final hidden state is extracted.
        - The last_hidden tensor now combines both forward and backward directions if enabled.
        """

        # utterance.size() = batch_size X seq_len
        utt_emb = self.embedding(utterance)  # utt_emb.size() = batch_size X seq_len X emb_size
        # adding dropout
        utt_emb = self.dropout(utt_emb)

        # pack_padded_sequence avoid computation over pad tokens reducing the computational cost
        packed_input = pack_padded_sequence(utt_emb, seq_lengths.cpu().numpy(),
                                           batch_first=True, enforce_sorted=False)  # NEW: enforce_sorted=False added

        # Process the batch
        packed_output, (last_hidden, cell) = self.utt_encoder(packed_input)

        # Unpack the sequence
        utt_encoded, _ = pad_packed_sequence(packed_output, batch_first=True)
        utt_encoded = self.dropout(utt_encoded)  # Dropout on LSTM output

        # Get the last hidden state
        if self.bidirectional:
            # NEW: For bidirectional LSTM, concatenate the last forward and backward hidden states
            # last_hidden.shape = [num_layers * num_directions, batch, hid_size]
            last_hidden = torch.cat((last_hidden[-2,:,:], last_hidden[-1,:,:]), dim=1)  # Combine forward + backward
        else:
            last_hidden = last_hidden[-1,:,:]  # Original behavior

        last_hidden = self.dropout(last_hidden)  # Dropout before intent prediction

        # Is this another possible way to get the last hiddent state? (Why?)
        # utt_encoded.permute(1,0,2)[-1]

        # Compute slot logits
        slots = self.slot_out(utt_encoded)  # [batch_size, seq_len, out_slot]
        # Compute intent logits
        intent = self.intent_out(last_hidden)  # [batch_size, out_int]

        # Slot size: batch_size, seq_len, classes
        slots = slots.permute(0,2,1)  # We need this for computing the loss (cross entropy)
        # Slot size: batch_size, classes, seq_len

        return slots, intent