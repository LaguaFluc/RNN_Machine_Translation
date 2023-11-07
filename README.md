# RNN_Machine_Translation
Using Tanslation data(en-fr) given by PyTorch Tutorial, my goal is to use the tutorial code to build a RNN model to translate English to French.


`prepare_data.py`: load and process language data
goal: get all the word information about input_lang and output_lang, also their pairs.

`train_prepare.py` process the data, and input it to the network.
goal: format sentence data, then ouput the tensor type of input_lang and output_lang.

`build_network.py` build a RNN network, simple, one layer RNN and a fully connected linear layer.

`train_network.py` define the method of train and evaluate, finally sample serveral pairs to see the sentence output.
goal: train model and evaluate model.