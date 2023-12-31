# https://blog.csdn.net/u013628121/article/details/114271540
# https://zhuanlan.zhihu.com/p/34418001

import torch
import torch.nn as nn
from build_network import Many2ManyRNN
from train_prepare import get_dataloader
from tempfile import TemporaryDirectory

# from torchtext.data.metrics import bleu_score
from torchmetrics.functional.text import bleu_score

import time
import os
import pathlib

import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))

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
            # seq_len <= MAX_LENGTH
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
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    cwd = os.getcwd()
    path_dir = pathlib.Path(cwd)
    best_model_params_path = path_dir / 'best_model_params.pt'

    for epoch in range(1, n_epochs + 1):
        loss = train_epoch(epoch, train_dataloader, model, optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if loss < best_val_loss:
            best_val_loss = loss
            torch.save(model.state_dict(), best_model_params_path)

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('(%s) (%d %d%%) %.4f' % (timeSince(start_time, epoch / n_epochs), epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
    model.load_state_dict(torch.load(best_model_params_path)) # load best model states
    return model, best_val_loss

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
    return " ".join(output_words)

def evaluateOneSentence(
    model: nn.Module, 
    input_sentence: str,
    target_sentence: str, 
    input_lang,
    output_lang,
    max_length: int = 10
    ):
    model.eval()
    # 在``sentence``中随机选择一个位置，用于输入
    print("----------")
    print("input_sentence: ", input_sentence, " target_sentence: ", target_sentence)
    output_words = sentence2sentence(model, input_sentence, input_lang, output_lang)

    bleuScore = bleu_score(
        [output_words],
        [target_sentence],
        n_gram=2
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
    model.eval()
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
        print("==============")
        pair = random.choice(pairs)
        print("input_sentence: ", pair[0])
        print("target_sentence: ", pair[1])
        output_words, bleuScore = evaluateOneSentence(
            model, pair[0], pair[1],
            input_lang, output_lang
            )
        output_str = output_words
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
    import yaml
    with open('config.yml', 'r', encoding='utf-8') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        SOS_token = config['SOS_token']
        EOS_token = config['EOS_token']
        MAX_LENGTH = config['MAX_LENGTH']
        batch_size = config['batch_size']

        num_layers = config['num_layers']

        log_interval = config['log_interval']
        print_every = config['print_every']
        plot_every = config['plot_every']

        n_epochs = config['n_epochs']
        hidden_size = config['hidden_size']
        learning_rate = config['learning_rate']
        seq_len = MAX_LENGTH

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # =======================
    # 1. 获取数据
    input_lang, output_lang, train_dataloader = get_dataloader(batch_size)
    vocab_size = input_lang.n_words

    # =======================
    # 2. 初始化模型
    # 在一个epoch中训练
    model = Many2ManyRNN(
        input_lang.n_words,
        hidden_size,
        hidden_size,
        dropout_p=0.1
    ).to(device)

    # model, best_val_loss = train(
    #     train_dataloader,
    #     model,
    #     n_epochs,
    #     learning_rate
    # )
    best_model_params_path = "best_model_params.pt"
    model.load_state_dict(torch.load(best_model_params_path))

    # ======================
    # 3. 评估模型（顺便测试一下）
    test_evaluateOneSentence = True
    test_evaluateNSentence = False
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
