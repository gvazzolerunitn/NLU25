# Add the class of your model only
# Here is where you define the architecture of your model using pytorch

import torch.nn as nn
    
# OLD CODE    
""" class LM_RNN(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_RNN, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)
        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.rnn = nn.RNN(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index
        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)
        rnn_out, _  = self.rnn(emb)lstm_out, _ = self.lstm(emb)
        output = self.output(rnn_out).permute(0,2,1)
        return output """
    
class LM_LSTM(nn.Module):
    def __init__(self, emb_size, hidden_size, output_size, pad_index=0, out_dropout=0.1,
                 emb_dropout=0.1, n_layers=1):
        super(LM_LSTM, self).__init__()
        # Token ids to vectors, we will better see this in the next lab 
        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_index)

        # ADDING DROPOUT AFTER THE EMBEDDING LAYER
        self.emb_dropout = nn.Dropout(emb_dropout)

        # Pytorch's RNN layer: https://pytorch.org/docs/stable/generated/torch.nn.RNN.html
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, bidirectional=False, batch_first=True)    
        self.pad_token = pad_index

        # ADDING DROPOUT BEFORE THE OUTPUT LAYER
        self.out_dropout = nn.Dropout(out_dropout)

        # Linear layer to project the hidden layer to our output space 
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_sequence):
        emb = self.embedding(input_sequence)

        # DROPOUT AFTER THE EMBEDDING LAYER
        emb = self.emb_dropout(emb) 

        lstm_out, _ = self.lstm(emb)

        # DROPOUT BEFORE THE OUTPUT LAYER
        lstm_out = self.out_dropout(lstm_out)

        output = self.output(lstm_out).permute(0,2,1)
        return output