#Author: Sayantan Dutta
#Program:  BOW_model.py
#part- 1b


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torch.distributed as dist

'''
We will now train another bag of words model but use the pre-trained GloVe features in place of the word embedding layer from before. Typically by leveraging larger datasets and regularization techniques, overfitting can be reduced. The GloVe features were pretrained on over 840 billion tokens. Our training dataset contains 20 million tokens and 2⁄3 of the 20 million tokens are part of the unlabeled reviews which weren’t used in part 1a. The GloVe dataset is over 100 thousand times larger.

The hope is then that these 300 dimensional GloVe features already contain a significant amount of useful information since they were pre-trained on such a large dataset and that will improve performance for our sentiment classification.
'''

class BOW_model(nn.Module):
    def __init__(self, no_of_hidden_units):
        super(BOW_model, self).__init__()

        self.fc_hidden1 = nn.Linear(300, no_of_hidden_units)
        self.bn_hidden1 = nn.BatchNorm1d(no_of_hidden_units)
        self.dropout1 = torch.nn.Dropout(p=0.5)

        # self.fc_hidden2 = nn.Linear(no_of_hidden_units,no_of_hidden_units)
        # self.bn_hidden2 = nn.BatchNorm1d(no_of_hidden_units)
        # self.dropout2 = torch.nn.Dropout(p=0.5)

        self.fc_output = nn.Linear(no_of_hidden_units, 1)

        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, x, t):
        h = self.dropout1(F.relu(self.bn_hidden1(self.fc_hidden1(x))))
        # h = self.dropout2(F.relu(self.bn_hidden2(self.fc_hidden2(h))))
        h = self.fc_output(h)

        return self.loss(h[:, 0], t), h[:, 0]
		
'''
This is actually simpler than part 1a considering it doesn’t need to do anything for the embedding layer. We know the input x will be a torch tensor of batch_size by 300 (the mean GloVe features over an entire sequence).
'''