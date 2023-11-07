import torch
import torch.nn as nn

MAX_LENGTH = 10
# TODO: 这是原来的网络，我看不明白为什么要这样做
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, input, hidden):
        input_combined = torch.cat((category, input, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class Many2ManyRNN(nn.Module):
    def __init__(self, vocab_size, input_size, hidden_size, dropout_p, num_layers=1):
        super(Many2ManyRNN, self).__init__()
        self.vocab_size = vocab_size
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(self.vocab_size, self.input_size)
        self.rnn = nn.RNN(self.input_size, self.hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        self.init_weights()

    def init_weights(self) ->None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        embedded = self.embedding(inputs)

        # print("embedded.shape", embedded.shape)
        h0 = self.initHidden(batch_size)
        # print("h0.shape: ", h0.shape)
        output, hidden = self.rnn(embedded, h0)
        output = self.linear(output)
        # print("output.shape: ", output.shape)
        return output

    def initHidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)

if __name__ == "__main__":
    import torch.nn.functional as F
    batch_size = 64
    vocab_size = 1000
    input_size = 32
    hidden_size = 16
    output_size = 32
    dropout_p = 0.1
    rnn = Many2ManyRNN(vocab_size, input_size, hidden_size, dropout_p)
    x = torch.randint(0, 200, (batch_size, MAX_LENGTH))
    print(x.shape)


    y = rnn(x)
    criterion = nn.NLLLoss()
    target = torch.randint(0, vocab_size, (batch_size, MAX_LENGTH))
    # 需要transpose y的最后两个维度
    dim_batch_size, dim_vocab_size = len(y.size()) - 2, len(y.size()) - 1
    loss = criterion(F.log_softmax(y, dim=-1).transpose(-2, -1), target)
    print("loss: ", loss)
    print(y.shape)
