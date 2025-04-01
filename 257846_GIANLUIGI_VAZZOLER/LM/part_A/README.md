Firstly, I started by implementing a LSTM instead of RNN as the first bullet point required. This was done by using the original LM_RNN code and replacing the RNN class with the LSTM one of torch.

This way, I ended up having the following results:
    -PPL => 152.318
    - Best PPL => 152.129
