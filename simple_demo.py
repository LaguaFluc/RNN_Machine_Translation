

import os
print(os.getcwd())

#%%
import torch.nn as nn
import torch
rnn = nn.RNN(10, 20, 2)
input = torch.randn(5, 3, 10)
h0 = torch.randn(2, 3, 20)
output, hn = rnn(input, h0)
output.size(), hn.size()


#%%
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
class SimpleCustomBatch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.inp = torch.stack(transposed_data[0], 0)
        self.tgt = torch.stack(transposed_data[1], 0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self

def collate_wrapper(batch):
    return SimpleCustomBatch(batch)

inps = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
tgts = torch.arange(10 * 5, dtype=torch.float32).view(10, 5)
dataset = TensorDataset(inps, tgts)

loader = DataLoader(dataset, batch_size=2, collate_fn=collate_wrapper,
                    pin_memory=True)

for batch_ndx, sample in enumerate(loader):
    print(batch_ndx, sample.inp.size(), sample.tgt.size())
    print(sample.inp.is_pinned())
    print(sample.tgt.is_pinned())


#%%
import torch
x = torch.arange(1., 6.)
print(x)
torch.topk(x, 1)
# %%

import torch.nn as nn
import torch

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs):
        embedded =  self.dropout(self.embedding(input))

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SOS_token = 0
EOS_token = 1
MAX_LENGTH = 10
hidden_size = 16
output_size = 1000
model = AttnDecoderRNN(hidden_size, output_size)
x = torch.randint(0, output_size, (1, MAX_LENGTH))
ouput, _, _ = model(x, x)


#%%
import torch
loss = torch.nn.NLLLoss()
input = torch.tensor([[ 1.1879,  1.0780,  0.5312],
        [-0.3499, -1.9253, -1.5725],
        [-0.6578, -0.0987,  1.1570]])
target = torch.tensor([0, 1, 2])
nllloss = loss(input, target)
print("nlllos: ", nllloss)

loss = torch.nn.CrossEntropyLoss()
target = torch.tensor([0, 1, 2])

our_softmax = torch.nn.Softmax(dim=1)(input)
print("softmax: ", our_softmax)
our_cce = torch.log(our_softmax)
print("log_softMax: ", our_cce)

log_sum = 0
for i in range(input.size(0)):
    input_idx = target[i]
    log_sum += our_cce[i][input_idx]
avg_log_sum = log_sum / input.size(0)


print("log_softmax + NLLLoss: ", avg_log_sum)
print("CrossEntropy: ", loss(input, target))

#%%
import torch
criterion = torch.nn.NLLLoss()
batch_size = 32
seq_len = 10
vocab_size = 100

output = torch.randn(batch_size, seq_len, vocab_size)
output = torch.log_softmax(output, dim=-1)
target = torch.randint(0, vocab_size, (bacth_size, seq_len))
print(output.size())
loss = criterion(output, target)
loss

#%%
import torch
import torch.nn as nn
m = nn.LogSoftmax(dim=1)
loss = nn.NLLLoss()
# input is of size N x C = 3 x 5
input = torch.randn(2, 5, 3, requires_grad=True)
# each element in target has to have 0 <= value < C
target = torch.tensor([[1, 0, 4], [1, 2, 3]])
output = loss(m(input), target)
output

#%%
# 2D loss example (used, for example, with image inputs)
N, C = 5, 4
loss = nn.NLLLoss()
# input is of size N x C x height x width
data = torch.randn(N, 16, 10, 10)
conv = nn.Conv2d(16, C, (3, 3))
m = nn.LogSoftmax(dim=1)
# each element in target has to have 0 <= value < C
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
output = loss(m(conv(data)), target)
print(m(conv(data)).shape, target.shape)
output.backward()


#%%
import torch
x = torch.randint(0, 10, (2, 3))

_, topi = torch.topk(x, 1)
topi.squeeze().shape

#%%
import torch
from torchtext.data.metrics import bleu_score

A = [" ".join(['SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS', 'SOS'])]
B = ['he s looking at you']
bleu_score(A, B)