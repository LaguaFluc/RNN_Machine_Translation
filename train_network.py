# https://blog.csdn.net/u013628121/article/details/114271540
# https://zhuanlan.zhihu.com/p/34418001

import torch
import torch.nn as nn
from build_network import Many2ManyRNN
from train_prepare import get_dataloader

from torchtext.data.metrics import bleu_score

import time

# 3. 开始训练
def train_epoch(
    epoch_idx,
    dataloader, 
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    criterion
    ) -> None:
    model.train() # turn on train mode

    start_time = time.time()
    total_loss = 0

    # 对数据集中一个个batch循环迭代
    for batch_idx, data in enumerate(dataloader):
        input_word_vector, out_word_vector = data

        output = model(input_word_vector)

        loss = criterion(output.transpose(-1, -2), out_word_vector)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if (batch_idx + 1) % log_interval == 0 and batch_idx > 0:
            ms_per_batch = (time.time() - start_time) * 1000 / log_interval
            cur_loss = total_loss / log_interval

            #  TODO:
            # calculate BLEU score
            # candidate: [batch_size, seq_len]
            # reference: [batch_size, seq_len]
            # bleu_score = bleu_score()
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch_idx, batch_idx * len(data), len(dataloader.dataset),
                log_interval * batch_idx / len(dataloader), loss.item()))
            total_loss = 0
            start_time = time.time()
    
    return total_loss / len(dataloader)

def train(
    train_dataloader,
    model: nn.Module, 
    n_epochs,
    learning_rate,
    print_every=100,
    plot_every=100
    ):
    start_time = time.time()
    plot_losses = []
    print_loss_total = 0
    plot_loss_total = 0

    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(epoch, train_dataloader, model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%s) (%d %d%%) %.4f' % (timeSince(start_time), epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



# =======================
# 4. 评估模型
# 1. 输入一个句子列表，输出一个预测的句子列表
# 2. 输入多个句子列表，并且输入对应的输出列表，计算BLEU分数
from typing import List
from train_prepare import tensorFromSentence
def sentence2sentence(
    model: nn.Module,
    input_sentence: str,
    input_lang,
    output_lang,
    ) -> List[str]:
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, input_sentence)
        output = model(input_tensor)
        # _, topi = output.top(1)
        _, topi = torch.topk(output, 1)
        output_ids = topi.squeeze()

        output_words = []
        for idx in output_ids:
            if idx.item() == EOS_token:
                output_words.append("<EOS>")
                break
            output_words.append(output_lang.index2word[idx.item()]) 
    return output_words

def evaluateOneSentence(
    model: nn.Module, 
    input_sentence: str,
    target_sentence: str, 
    input_lang,
    output_lang,
    max_length: int = 10
    ):
    # 在``sentence``中随机选择一个位置，用于输入
    
    output_words = sentence2sentence(model, input_sentence, input_lang, output_lang)

    bleuScore = bleu_score(
        [output_words],
        [target_sentence]
    )
    return output_words, bleuScore

def evaluateNSentence(
    model: nn.Module,
    input_sentences: List[str],
    target_sentences: List[str],
    input_lang, 
    output_lang,
    max_length: int = 10
    ):
    output_sentences = []
    for input_sentence, target_sentence in zip(input_sentences, target_sentences):
        output_sentence, bleuScore = evaluateOneSentence(
            model, input_sentence, target_sentence, input_lang, output_lang, max_length
        )
        output_sentences.append(output_sentence)

    bleuScore = bleu_score(output_sentences, target_sentences)
    return output_sentences, bleuScore
# 5. 输入一个英语句子，输出法语预测

def testEvaluateOneSentence():
    for i in range(5):
        pair = random.choice(pairs)
        print("input_sentence: ", pair[0])
        print("target_sentence: ", pair[1])
        output_words, bleuScore = evaluateOneSentence(
            model, pair[0], pair[1],
            input_lang, output_lang
            )
        output_str = " ".join(output_words)
        print("output_sentence: ", output_str)
        print("bleuScore: ", bleuScore)

def testEvaluateNSentence():
    # 测试evaluateNSentence
    # (a) 生成随机句子
    choice_pairs = random.choices(pairs, k=5)
    input_sentences = [pair[0] for pair in choice_pairs]
    target_sentences = [pair[1] for pair in choice_pairs]

    output_sentences, bleuScore = evaluateNSentence(
        model,
        input_sentences,
        target_sentences,
        input_lang,
        output_lang
    )
    print("output_sentences: ", output_sentences)
    print("bleu_score: ", bleuScore)

if __name__ == "__main__":
    # =======================
    # 0. 初始化数据
    SOS_token = 0
    EOS_token = 1

    num_layers = 1
    MAX_LENGTH = 10

    log_interval = 200
    print_every = 5
    plot_every = 5

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =======================
    # 1. 获取数据
    batch_size = 32
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    vocab_size = input_lang.n_words

    # =======================
    # 2. 初始化模型
    # 在一个epoch中训练
    hidden_size = 128
    batch_size = 32
    seq_len = MAX_LENGTH

    n_epochs = 1
    learning_rate = 1e-2

    model = Many2ManyRNN(
        input_lang.n_words,
        hidden_size,
        hidden_size,
        dropout_p=0.1
    ).to(device)

    train(
        train_dataloader,
        model,
        n_epochs,
        learning_rate
    )

    # ======================
    # 3. 评估模型（顺便测试一下）
    test_evaluateOneSentence = False
    test_evaluateNSentence = True
    # 测试输出evalueOneSentence
    import random
    from prepare_data import prepareData
    input_lang, output_lang, pairs = prepareData('eng', 'fra', True)
    if test_evaluateOneSentence:
        testEvaluateOneSentence()

    # 测试evaluateNSentence
    # (a) 生成随机句子
    if test_evaluateNSentence:
        testEvaluateNSentence()
    # =======================
    # 3. 开始训练
    # 4. 评估模型
    # 5. 输入一个英语句子，输出法语预测
