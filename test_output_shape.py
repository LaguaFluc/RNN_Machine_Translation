import torch
import torch.nn as nn
import torchtext
from torchtext.vocab import Vectors
import numpy as np
import random

USE_CUDA = torch.cuda.is_available()

random.seed(53113)
np.random.seed(53113)
torch.manual_seed(53113)
if USE_CUDA:
    torch.cuda.manual_seed(53113)

BATCH_SIZE = 32 
EMBEDDING_SIZE = 500  
MAX_VOCAB_SIZE = 50000  


# - 使用text8作为我们的训练，验证和测试数据
# - torchtext提供了LanguageModelingDataset这个class来帮助我们处理语言模型数据集
# - BPTTIterator可以连续地得到连贯的句子


TEXT = torchtext.data.Field(lower=True)   #Field对象：如何预处理文本数据的信息，这里定义单词全部小写
# torchtext提供了LanguageModelingDataset这个class来帮助我们处理语言模型数据集
train, val, test = torchtext.datasets.LanguageModelingDataset.splits(
                    path=".",  #当前文件夹
                    train="text8.train.txt", 
                    validation="text8.dev.txt", 
                    test="text8.test.txt", 
                    text_field=TEXT)

TEXT.build_vocab(train, max_size=MAX_VOCAB_SIZE)
# build_vocab可以根据我们提供的训练数据集来创建最高频单词的单词表，max_size帮助我们限定单词总量。
print("vocabulary size: {}".format(len(TEXT.vocab)))


print(TEXT.vocab.itos[0:50]) 
# 这里越靠前越常见，增加x了两个特殊的token，<unk>表示未知的单词，<pad>表示padding。
print("------"*10)
print(list(TEXT.vocab.stoi.items())[0:50])


VOCAB_SIZE = len(TEXT.vocab) # 50002
train_iter, val_iter, test_iter = torchtext.data.BPTTIterator.splits(
                            (train, val, test), 
                            batch_size=BATCH_SIZE, 
                            device=-1, 
                            bptt_len=50, # 反理解为一个样本有多少个单词传入模型
                            repeat=False, 
                            shuffle=True)
# BPTTIterator可以连续地得到连贯的句子，BPTT的全称是back propagation through time
print(type(train_iter))