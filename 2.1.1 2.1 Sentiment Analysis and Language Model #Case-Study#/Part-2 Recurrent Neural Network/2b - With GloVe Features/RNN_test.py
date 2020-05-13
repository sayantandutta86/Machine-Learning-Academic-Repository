#Author: Sayantan Dutta
#Program:  RNN_test.py
#Assignment 5
#part- 2b

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



# Hyperparameters

batch_size = 200
no_of_epochs = 10
vocab_size = 8000

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

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

        x_input2 = [x_test[j] for j in I_permutation[i:i+batch_size]]

        # sequence_length = 100
        # sequence_length = sequence_lengths[1]
        sequence_length = (epoch + 1) * 50

        x_input = np.zeros((batch_size,sequence_length),dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if(sl < sequence_length):
                x_input[j,0:sl] = x
            else:
                start_index = np.random.randint(sl-sequence_length+1)
                x_input[j,:] = x[start_index:(start_index+sequence_length)]
        x_input = glove_embeddings[x_input]
        y_input = y_test[I_permutation[i:i+batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
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
    epoch_loss /= (epoch_counter/batch_size)

    test_accu.append(epoch_acc)

    time2 = time.time()
    time_elapsed = time2 - time1

    print("sequence length: ",sequence_length, "test accuracy: ","%.2f" % (epoch_acc*100.0), "test loss: ","%.4f" % epoch_loss,
          "time taken: ","%.4f" % float(time_elapsed))
		  
'''

Here is an example output from one of the models I trained.

50 78.94 0.5010 24.4657
100 84.63 0.3956 48.4858
150 87.62 0.3240 75.0972
200 89.18 0.2932 109.8104
250 89.90 0.2750 149.9544
300 90.50 0.2618 193.7136
350 90.83 0.2530 245.5717
400 90.90 0.2511 299.2605
450 91.08 0.2477 360.4909
500 91.19 0.2448 428.9970

We see some progress here and the results are the best yet. Although the bag of words model with GloVe features performed poorly, we now see the GloVe features are much better utilized when allowing the model to handle the temporal information and overfitting seems to be less of a problem since we aren’t retraining the embedding layer. As we saw in part 2a, the LSTM seems to be overkill and the dataset doesn’t seem to be complex enough to train a model completely from scratch without overfitting.

'''