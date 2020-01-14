#Author: Sayantan Dutta
#Program:  RNN_model.py
#part- 2a

'''
Take the following two reviews: * “Although the movie had great visual effects, I thought it was terrible.” * “Although the movie had terrible visual effects, I thought it was great.”

The first review clearly has an overall negative sentiment while the bottom review clearly has an overall positive sentiment. Both sentences would result in the exact same output if using the bag of words approach.

Clearly there is a lot of useful information which could maybe be utilized more effectively if we didn’t discard the sequence information like we did in part 1. By designing a model capable of capturing this additional source of information, we can potentially achieve better results but also greatly increase the risk of overfitting. This is heavily related to the curse of dimensionality.

A recurrent neural network can be used to maintain the temporal information and process the data as an actual sequence. Part 2 will consist of training recurrent neural networks built with LSTM layers. We will train two separate models again, one from scratch with a word embedding layer and one with GloVe features.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

'''An LSTM Cell in pytorch needs a variable for both the internal cell state c(t) as well as the hidden state h(t). Let’s start by creating a StatefulLSTM() class that can maintain these values for us.'''

class StatefulLSTM(nn.Module):
    def __init__(self, in_size, out_size):
        super(StatefulLSTM, self).__init__()

        self.lstm = nn.LSTMCell(in_size, out_size)
        self.out_size = out_size

        self.h = None
        self.c = None

    def reset_state(self):
        self.h = None
        self.c = None

    def forward(self, x):
        batch_size = x.data.size()[0]
        if self.h is None:
            state_size = [batch_size, self.out_size]
            self.c = Variable(torch.zeros(state_size)).cuda()
            self.h = Variable(torch.zeros(state_size)).cuda()
        self.h, self.c = self.lstm(x, (self.h, self.c))

        return self.h

'''This module uses self.h and self.c to maintain the necessary information while processing a sequence. We can reset the layer at anytime by calling reset_state(). The nn.LSTMCell() contains all of the actual LSTM weights as well as all of the operations.

When processing sequence data, we will need to apply dropout after every timestep. It has been shown to be more effective to use the same dropout mask for an entire sequence as opposed to a different dropout mask each time. More details as well as a paper reference can be found here. Pytorch doesn’t have an implementation for this type of dropout so we will make it ourselves.'''


class LockedDropout(nn.Module):
    def __init__(self):
        super(LockedDropout,self).__init__()
        self.m = None

    def reset_state(self):
        self.m = None

    def forward(self, x, dropout=0.5, train=True):
        if train==False:
            return x
        if(self.m is None):
            self.m = x.data.new(x.size()).bernoulli_(1 - dropout)
        mask = Variable(self.m, requires_grad=False) / (1 - dropout)

        return mask * x

'''Note that if this module is called with train set to False, it will simply return the exact same input. If train is True, it checks to see if it already has a dropout mask self.m. If it does, it uses this same mask on the data. If it doesn’t, it creates a new mask and stores it in self.m. As long as we reset our LockedDropout() layer at the beginning of each batch, we can have a single mask applied to the entire sequence.'''



class RNN_model(nn.Module):
    def __init__(self,vocab_size,no_of_hidden_units):
        super(RNN_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size,no_of_hidden_units)#,padding_idx=0)

        self.lstm1 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        self.bn_lstm1= nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = LockedDropout() #torch.nn.Dropout(p=0.5)

        # self.lstm2 = StatefulLSTM(no_of_hidden_units,no_of_hidden_units)
        # self.bn_lstm2= nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = LockedDropout() #torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.BCEWithLogitsLoss()

    def reset_state(self):
        self.lstm1.reset_state()
        self.dropout1.reset_state()
        # self.lstm2.reset_state()
        # self.dropout2.reset_state()

    def forward(self, x, t, train=True):

        embed = self.embedding(x) # batch_size, time_steps, features

        no_of_timesteps = embed.shape[1]

        self.reset_state()

        outputs = []
        for i in range(no_of_timesteps):

            h = self.lstm1(embed[:,i,:])
            h = self.bn_lstm1(h)
            h = self.dropout1(h,dropout=0.5,train=train)

            # h = self.lstm2(h)
            # h = self.bn_lstm2(h)
            # h = self.dropout2(h,dropout=0.3,train=train)

            outputs.append(h)

        outputs = torch.stack(outputs) # (time_steps,batch_size,features)
        outputs = outputs.permute(1,2,0) # (batch_size,features,time_steps)

        pool = nn.MaxPool1d(no_of_timesteps)
        h = pool(outputs)
        h = h.view(h.size(0),-1)
        #h = self.dropout(h)

        h = self.fc_output(h)

        return self.loss(h[:,0],t), h[:,0]#F.softmax(h, dim=1)

'''We create an embedding layer just like in part 1a. We can create a StatefulLSTM() layer we created above as well as a LockedDropout() layer. The reset_state() function is used so we can easily reset the state of any layer in our model that needs it.

The forward() function has a few new features. First notice we have an additional input variable train. We can’t rely on model.eval() to handle dropout appropriately for us anymore so we will need to do it ourselves by passing this variable to the LockedDropout() layer.

Assume our input x is batch_size by sequence_length. This means we will be training everything with a fixed sequence length (we will go over how this is done in the actual training loop). By passing this tensor into the embedding layer, it will return an embedding for every single value meaning embed is batch_size by sequence_length by no_of_hidden_units. We then loop over the sequence one step at a time and append h to the outputs list every time.

The list outputs is converted to a torch tensor using torch.stack() and we transform the tensor to get back the shape batch_size by no_of_hidden_units by no_of_timesteps. We need it in this ordering because the nn.MaxPool1d() operation pools over the last dimension of a tensor. After pooling, the h.view() operation removes the last dimension from the tensor (it’s only length 1 after the pooling operation) and we’re left with a vector h of size batch_size by no_of_hidden_units. We now pass this into the output layer.

This is a good time to mention how there isn’t actually a whole lot of difference between what we’re doing here and what we did with the bag of words model. The mean embedding over an entire sequence can be thought of as simply a mean pooling operation. It just so happens that we did this mean pooling operation on a bunch of separate words without a word knowing anything about its surrounding context. We pooled before processing any of the temporal information.

This time we are processing the word embeddings in order such that each subsequent output actually has the context of all words preceding it. Eventually we still need to get to a single output for the entire sequence (positive or negative sentiment) and we do this by now using a max pooling operation over the number of timesteps. As opposed to the bag of words technique, we pooled after processing the temporal information.

We could use average pooling instead of max pooling if we wanted. Max pooling seems to work a little better in practice but it depends on the dataset and problem. Intuitively one can imagine the recurrent network processing a very long sequence of short phrases. These short phrases can appear anywhere within the long sequence and carry some important information about the sentiment. After each important short phrase is seen, it outputs a vector with some large values. Eventually all of these vectors are pooled where only the max values are kept. You are left with a vector for the entire sequence containing only the important information related to sentiment as all of the unimportant information was thrown out. For classifying a sequence with a single label, we will always have to eventually collapse the temporal information. The decision in relation to network design deals with where within the model this should be done.'''