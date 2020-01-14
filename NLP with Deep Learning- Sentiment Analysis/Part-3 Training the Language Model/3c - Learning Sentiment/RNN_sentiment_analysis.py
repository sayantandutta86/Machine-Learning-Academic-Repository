# Author - Sayantan Dutta
# Program: RNN_sentiment_analysis.py
# Assignment 6
# Assignment- 3c

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

from RNN_model import RNN_model
from RNN_language_model import RNN_language_model

resume_training = False

# Hyperparameters.

LR = 0.001
no_of_epochs = 20
batch_size = 200
vocab_size = 8000
num_hidden_units= 500 # start off with 500, try 300 too
optimizer = 'adam'
seq_len_train = 100
seq_len_test = 400


"""
I’d suggest trying to train a model on short sequences (50 or less) 
as well as long sequences (250+) just to see the difference 
in its ability to generalize.
Also try different hidden units
"""

sequence_lengths = [seq_len_train, seq_len_test]


# Load training data
x_train = []
with io.open('../preprocessed_data/imdb_train.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0

    x_train.append(line)

# Only the first 25,000 are labeled
x_train = x_train[0:25000]

# The first 125000 are labeled 1 for positive and the last 12500 are labeled 0 for negative
y_train = np.zeros((25000,))
y_train[0:12500] = 1

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

# model = RNN_model(vocab_size, num_hidden_units)
# model.cuda()
if resume_training:
    print('Resume Training..')
    model = torch.load('rnn.model')
else:
    print('Start Training from beginning..')
    model = RNN_model(vocab_size, num_hidden_units)
model.cuda()
language_model = torch.load('language.model')

# Copy the weights for the embedding and LSTM layers from the language model
model.embedding.load_state_dict(language_model.embedding.state_dict())
model.lstm1.lstm.load_state_dict(language_model.lstm1.lstm.state_dict())
model.bn_lstm1.load_state_dict(language_model.bn_lstm1.state_dict())
model.lstm2.lstm.load_state_dict(language_model.lstm2.lstm.state_dict())
model.bn_lstm2.load_state_dict(language_model.bn_lstm2.state_dict())
model.lstm3.lstm.load_state_dict(language_model.lstm3.lstm.state_dict())
model.bn_lstm3.load_state_dict(language_model.bn_lstm3.state_dict())
model.cuda()

params = []
# for param in model.embedding.parameters():
#     params.append(param)
# for param in model.lstm1.parameters():
#     params.append(param)
# for param in model.bn_lstm1.parameters():
#     params.append(param)
# for param in model.lstm2.parameters():
#     params.append(param)
# for param in model.bn_lstm2.parameters():
#     params.append(param)
for param in model.lstm3.parameters():
    params.append(param)
for param in model.bn_lstm3.parameters():
    params.append(param)
for param in model.fc_output.parameters():
    params.append(param)

if optimizer == 'adam':
    optimizer = optim.Adam(params, lr=LR)
elif optimizer == 'sgd':
    optimizer = optim.SGD(params, lr=LR, momentum=0.9)
	
'''Before defining the optimizer, we’re going to make a list of parameters we want to train. The model will overfit if we train everything. However, we can choose to just fine-tune the last LSTM layer and the output layer.'''

L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []

print("Begin training...")

for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i + batch_size]]

        # sequence_length = 100
        sequence_length = sequence_lengths[0]

        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if (sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl - sequence_length + 1)
                x_input[j, :] = x[start_index:(start_index + sequence_length)]
        y_input = y_train[I_permutation[i:i + batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data, target, train=True)
        loss.backward()

        optimizer.step()  # update weights

        prediction = pred >= 0.0
        truth = target >= 0.5
        acc = prediction.eq(truth).sum().cpu().data.numpy()
        epoch_acc += acc
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter / batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print("epoch:", epoch,
          "training accuracy: " ,"%.2f" % (epoch_acc * 100.0),
          "training loss: ","%.4f" % epoch_loss,
          "elpased time:","%.4f" % float(time.time() - time1))

    if (epoch + 1) % 5 == 0:
        torch.save(model, 'rnn.model')
        print("Model Saved...")

    ### test
    if (epoch + 1) % 3 == 0:

        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            x_input2 = [x_test[j] for j in I_permutation[i:i + batch_size]]

            # sequence_length = 100
            sequence_length = sequence_lengths[1]

            x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if (sl < sequence_length):
                    x_input[j, 0:sl] = x
                else:
                    start_index = np.random.randint(sl - sequence_length + 1)
                    x_input[j, :] = x[start_index:(start_index + sequence_length)]
            y_input = y_train[I_permutation[i:i + batch_size]]

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

        print("  ","Epoch:" , epoch, "Epoch test accuracy:","%.2f" % (epoch_acc * 100.0), "Epoch loss:","%.4f" % epoch_loss,
              "Time elapsed: ","%.4f" % float(time_elapsed))



torch.save(model, 'rnn.model')
print("Model saved...")
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('rnn_data.npy', data)

'''
Here is an example output of mine for a particular model:

50 80.98 0.4255 17.0083
100 87.17 0.3043 30.2916
150 90.18 0.2453 45.0554
200 91.15 0.2188 59.9038
250 91.96 0.2022 74.8118
300 92.34 0.1960 89.7251
350 92.64 0.1901 104.7904
400 92.83 0.1863 119.8761
450 92.95 0.1842 134.8656
500 93.01 0.1828 150.3047
This performs better than all of the previous models. By leveraging the additional unlabeled data and pre-training the network as a language model, we can achieve even better results than the GloVe features trained on a much larger dataset. To put this in perspective, the state of the art on the IMDB sentiment analysis is around 97.4%.

'''
