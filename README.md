# RNN_Machine_Translation
Using Tanslation data(en-fr) given by PyTorch Tutorial, my goal is to use the tutorial code to build a RNN model to translate English to French.

Dataset: PyTorch Tutorial provided `data.zip`
Task: Machine Translation
Model: RNN + 1-layer Linear
Loss: `nn.CrossEntroy()`
Evaluation: BLEU_score(`torchtext.data.metrics`)

Main files introduction:
`prepare_data.py`: load and process language data
goal: get all the word information about input_lang and output_lang, also their pairs.

`train_prepare.py` process the data, and input it to the network.
goal: format sentence data, then ouput the tensor type of input_lang and output_lang.

`build_network.py` build a RNN network, simple, one layer RNN and a fully connected linear layer.

`train_network.py` define the method of train and evaluate, finally sample serveral pairs to see the sentence output.
goal: train model and evaluate model.


TODO:
- [ ] output many words one by one
- [ ] split dataset into train, val, test seperately.
- [ ] learn how to choose a loss function, according to different model output.

references:
https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html#
https://pytorch.org/tutorials/intermediate/char_rnn_generation_tutorial.html
https://blog.csdn.net/u013628121/article/details/114271540