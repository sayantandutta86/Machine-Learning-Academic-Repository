#Author: Sayantan Dutta
#Program:  BOW_model.py
#part-1a

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

'''

The inputs for defining our model object are vocab_size and no_of_hidden_units (we can decide on actual values for these later). We will need to define an embedding layer, a single hidden layer with batch normalization, an output layer, a dropout layer, and the loss function.



'''

class BOW_model(nn.Module):
    def __init__(self, vocab_size, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.embedding = nn.Embedding(vocab_size, no_of_hidden_units)

        self.fc_hidden = nn.Linear(no_of_hidden_units, no_of_hidden_units)
        self.bn_hidden = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()
		
'''
Mathematically speaking, there is nothing unique about a word embedding layer. It’s a matrix multiplication with a bias term. A word embedding layer actually has the exact same number of weights as a linear layer. The difference comes down to the PyTorch implementation. 
The input to the embedding layer is just an index while a linear layer requires an appropriately sized vector. 

We could use a linear layer if we wanted but we’d have to convert our token IDs to a bunch of 1-hot vectors and do a lot of unnecessary matrix multiplication since most of the entries are 0. As mentioned above, a matrix multiplying a 1-hot vector is the same as just extracting a single column of the matrix, which is a lot faster. 

Embedding layers are used frequently in NLP since the input is almost always a 1-hot representation.

Even though we will never explicitly see a 1-hot vector in the code, it is very useful intuitively to view the input as a 1-hot vector.
'''		

    def forward(self, x, t):
        bow_embedding = []
        for i in range(len(x)):
            lookup_tensor = Variable(torch.LongTensor(x[i])).cuda()
            embed = self.embedding(lookup_tensor)
            embed = embed.mean(dim=0)
            bow_embedding.append(embed)
        bow_embedding = torch.stack(bow_embedding)

        h = self.dropout(F.relu(self.bn_hidden(self.fc_hidden(bow_embedding))))
        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]

'''
When we call our model, we will need to provide it with an input x and target labels t.

This implementation is not typical in the sense that a for loop is used for the embedding layer as opposed to the more typical batch processing. Assume the input x is a list of length batch_size and each element of this list is a numpy array containing the token ids for a particular sequence. These sequences are different length which is why x is not simply a torch tensor of size batch_size by sequence_length. Within the loop, the lookup_tensor is a single sequence of token ids which can be fed into the embedding layer. This returns a torch tensor of length sequence_length by embedding_size. We take the mean over the dimension corresponding to the sequence length and append it to the list bow_embedding. This mean operation is considered the bag of words. Note this operation returns the same vector embed regardless of how the token ids were ordered in the lookup_tensor.

The torch.stack() operation is a simple way of converting a list of torch tensors of length embedding_size to a single torch tensor of length batch_size by embedding_size. The bow_embedding is passed through a linear layer, a batch normalization layer, a ReLU operation, and dropout.

Our output layer has only one output even though there are two classes for sentiment (positive and negative). We could’ve had a separate output for each class but I actually saw an improvement of around 1-2% by using the nn.BCEWithLogitsLoss() (binary cross entropy with logits loss) as opposed to the usual nn.CrossEntropyLoss() we usually see with multi-class classification problems.

Note that two values are returned, the loss and the logit score. The logit score can be converted to an actual probability by passing it through the sigmoid function or it can be viewed as any score greater than 0 is considered positive sentiment. The actual training loop ultimately determines what will be done with these two returned values.
'''