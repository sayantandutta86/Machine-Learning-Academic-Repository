#Author: Sayantan Dutta
#Program:  RNN_test.py
#part- 2a

'''These typically take longer to train so we don’t want to be testing on the full sequences after every epoch as it’s a huge waste of time. It’s easier and quicker to just have a separate script to test the model as we only need a rough idea of test performance during training.'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

import time
import os
import sys
import io
import argparse

from RNN_model import RNN_model

#Hyperparamters

vocab_size = 8000
batch_size = 200
no_of_epochs = 10

# Load testing data
x_test = []
with io.open('../preprocessed_data/imdb_test.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0

    x_test.append(line)

y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = torch.load('rnn.model')
model.cuda()

L_Y_test = len(y_test)

test_accu = []

for epoch in range(no_of_epochs):

    # Test

    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):

        x_input2 = [x_test[j] for j in I_permutation[i:i + batch_size]]

        sequence_length = (epoch + 1) * 50

		'''
		
		This way it will loop through the entire test dataset and report results for varying sequence lengths. Here is an example I tried for a network trained on sequences of length 100 with 300 hidden units for the LSTM and a dictionary size of 8000 (+1 for unknown token).

		sequence length, test accuracy, test loss, elapsed time
		50  76.14 0.7071 17.5213
		100  81.74 0.5704 35.1576
		150  84.51 0.4760 57.9063
		200  86.10 0.4200 84.7308
		250  86.69 0.3985 115.8944
		300  86.98 0.3866 156.6962
		350  87.00 0.3783 203.2236
		400  87.25 0.3734 257.9246
		450  87.22 0.3726 317.1263
		
		The results here might go against the intuition of what was expected. It actually performs worse than the bag of words model from part 1a. However, this does sort of make sense considering we were already overfitting before and we vastly increased the capabilities of the model. It performs nearly as well and with enough regularization/dropout, you can almost certainly achieve better results. Training on shorter sequences can also potentially help to prevent overfitting.
		'''



        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if (sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl - sequence_length + 1)
                x_input[j, :] = x[start_index:(start_index + sequence_length)]
        y_input = y_test[I_permutation[i:i + batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data, target, train=False)

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter / batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("sequence length, test accuracy, test loss, eplased time")
    print(sequence_length, "%.2f" % (epoch_acc * 100.0), "%.4f" % epoch_loss,
          "%.4f" % float(time_elapsed))