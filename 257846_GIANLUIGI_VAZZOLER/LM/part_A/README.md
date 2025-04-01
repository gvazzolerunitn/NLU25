Firstly, I started by implementing a LSTM based model instead of a RNN one, as the first bullet point required. This was done by using the original LM_RNN code and replacing the RNN class with the LSTM one of torch.

This way, I ended up having the following results:
- PPL => 
- Best PPL => 

The performance is already compliant with the mandatory requirement of PPL < 250, but there still were other two modifications to do. Therefore, I then added two dropout layers, the first one had to be after the embedding layer, whereas the second one before the last linear layer.
The results of this further implementations showed some improvements, as:
- PPL =>
- Best PPL =>

The last step involved the changing of the optimizer, so I replaced SGD with AdamW. I also needed to change the learning rate, as Adam needs a pretty low value (the default one is 0.001). Hence the results here were:
- PPL =>
- Best PPL =>
