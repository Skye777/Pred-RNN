from __future__ import print_function

import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter
import prepare_data
from PredRnn import *
import time
import math
import random

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

MAX_LENGTH = 10
SOS_token = 0
EOS_token = 1


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)


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


def train(input_variable, target_variable, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion,
          max_length=MAX_LENGTH, teacher_forcing_ratio=0.5):
    encoder_hidden = encoder.initHidden()  # (1, 1, hidden_size)

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]  # lags
    target_length = target_variable.size()[0]  # steps

    loss = 0.0

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)    # input_variable[ei]=>(batch_size,h, w)
        encoder_outputs[ei] = encoder_output[0][0]

    decoder_input = Variable(torch.LongTensor([[SOS_token]]))
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    if use_teacher_forcing:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, target_variable[di])
            decoder_input = target_variable[di]

    else:
        for di in range(target_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)

            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            decoder_input = Variable(torch.LongTensor([[ni]]))
            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, target_variable[di])
            if ni == EOS_token:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / target_length


def trainIters(encoder, decoder, n_iters, data_generator, print_every=1000, plot_every=100, learning_rate=0.01):
    start_time = time.time()

    encoder_optimizer = t.optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = t.optim.SGD(decoder.parameters(), lr=learning_rate)

    # training_pairs = [variablesFromPair(random.choice(pairs)) for i in range(n_iters)]

    criterion = nn.NLLLoss()

    print_loss_total = 0
    plot_loss_total = 0
    plot_losses = list()
    for i in range(1, n_iters + 1):
        training_pair = next(data_generator)
        input_variable = training_pair[0]   # (lags, batch_size, h, w)
        target_variable = training_pair[1]

        loss = train(input_variable, target_variable, encoder,
                     decoder, encoder_optimizer, decoder_optimizer, criterion)
        print_loss_total += loss
        plot_loss_total += loss

        if i % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start_time, i / n_iters),
                                         i, i / n_iters * 100, print_loss_avg))

        if i % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0

    showPlot(plot_losses)


# def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
#     input_variable = variableFromSentence(input_lang, sentence)
#     input_length = input_variable.size()[0]
#     encoder_hidden = encoder.initHidden()
#
#     encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
#     encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
#
#     for ei in range(input_length):
#         encoder_output, encoder_hidden = encoder(input_variable[ei],
#                                                  encoder_hidden)
#         encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
#
#     decoder_input = Variable(torch.LongTensor([[SOS_token]]))  # SOS
#     decoder_input = decoder_input.cuda() if use_cuda else decoder_input
#
#     decoder_hidden = encoder_hidden
#
#     decoded_words = []
#     decoder_attentions = torch.zeros(max_length, max_length)
#
#     for di in range(max_length):
#         decoder_output, decoder_hidden, decoder_attention = decoder(
#             decoder_input, decoder_hidden, encoder_outputs)
#         decoder_attentions[di] = decoder_attention.data
#         topv, topi = decoder_output.data.topk(1)
#         ni = topi[0][0]
#         if ni == EOS_token:
#             decoded_words.append('<EOS>')
#             break
#         else:
#             decoded_words.append(output_lang.index2word[ni])
#
#         decoder_input = Variable(torch.LongTensor([[ni]]))
#         decoder_input = decoder_input.cuda() if use_cuda else decoder_input
#
#     return decoded_words, decoder_attentions[:di + 1]


use_cuda = True
n_layer = 4
batch_size = 128
hidden_size = [32, 32, 32, 32]
input_size = 3
height = 36
width = 80
lags = 12
steps = 12

file_path = 'sst.mon.mean1850-2015.nc'  #####

input, target = prepare_data.load_data(file_path, lags, steps)

data_generator = prepare_data.get_batches(input, target, batch_size)

# encoder1 = Encoder(input_lang.n_words, hidden_size)
encoder1 = Encoder(n_layers=n_layer, hidden_sizes=hidden_size, input_sizes=input_size,
                   batch_size=batch_size, height=height, width=width, use_cuda=use_cuda)
# decoder1 = Decoder(hidden_size, output_lang.n_words, dropout_p=0.1)
decoder1 = Decoder(n_layers=n_layer, hidden_sizes=hidden_size, input_sizes=input_size,
                   batch_size=batch_size, height=height, width=width, use_cuda=use_cuda)

if use_cuda:
    encoder1 = encoder1.cuda()
    attn_decoder1 = decoder1.cuda()

trainIters(encoder1, decoder1, 75000, data_generator, print_every=5000)

# if __name__ == '__main__':
#     main()
