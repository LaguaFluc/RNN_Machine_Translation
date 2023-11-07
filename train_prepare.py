
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
from prepare_data import prepareData


MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =========================
# Prepare training data

# sentence : [id1, id2, id3, ...]
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

# tensors: wordtensor, row vector, size: (1, 
# TODO : add EOS, using config files
def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

# batch_size here is the number of samples
def get_dataloader(batch_size):
    # input_lang: List[str], output_lang: List[str], pairs: List[List[str]]
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)

    # length of dataset
    n = len(pairs)
    # matrix, size: (n, MAX_LENGTH)
    # row: each sentence, 
    # col: each word in a sample sentence
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    # fill data
    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader


# ===========================
# test get_dataloader
if __name__ == "__main__":
    batch_size = 32
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

    for batch_ndx, sample in enumerate(train_dataloader):
        input_tensor, target_tensor = sample
        print(input_tensor.shape, target_tensor.shape)
        if (batch_ndx >= 1): break
