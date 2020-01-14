#Author: Sayantan Dutta
#Program:  BOW_sentiment_analysis.py
#part- 1b

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import time
import io

#Hyper-paramters
arg_vocab_size = 100000
arg_hidden_units = 500
arg_opt = 'adam'
arg_LR = 0.001
arg_batch_size = 200
arg_no_of_epochs = 6


from BOW_model import BOW_model

glove_embeddings = np.load('../preprocessed_data/glove_embeddings.npy')
vocab_size = arg_vocab_size

x_train = []
with io.open('../preprocessed_data/imdb_train_glove.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0
    line = line[line != 0]

    line = np.mean(glove_embeddings[line], axis=0)

    x_train.append(line)
x_train = np.asarray(x_train)
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

x_test = []
with io.open('../preprocessed_data/imdb_test_glove.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line, dtype=np.int)

    line[line > vocab_size] = 0
    line = line[line != 0]

    line = np.mean(glove_embeddings[line], axis=0)

    x_test.append(line)
x_test = np.asarray(x_test)
y_test = np.zeros((25000,))
y_test[0:12500] = 1

vocab_size += 1

model = BOW_model(arg_hidden_units)  # try 300 as well

model.cuda()

'''

This first part is nearly the same besides the fact that we can actually go ahead and do the mean operation for the entire sequence one time when loading in the data. We load in the glove_embeddings matrix, convert all out-of-dictionary tokens to the unknown token for each review, extract the embedding for each token in the sequence from the matrix, take the mean of these emeddings, and append this to the x_train or x_test list.

The rest of the code is the same besides grabbing the data for each batch within the actual train/test loop.

'''

##########

# optimizer = 'sgd'
# LR = 0.01
opt = arg_opt
LR = arg_LR
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)

batch_size = arg_batch_size
no_of_epochs = arg_no_of_epochs
L_Y_train = len(y_train)
L_Y_test = len(y_test)

model.train()

train_loss = []
train_accu = []
test_accu = []

for epoch in range(no_of_epochs):

    # training
    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_train)

    for i in range(0, L_Y_train, batch_size):


        x_input = x_train[I_permutation[i:i + batch_size]]
        y_input = y_train[I_permutation[i:i + batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data, target)
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

    print('Epoch:', epoch, 'Training Accuracy: ', "%.2f" % (epoch_acc * 100.0), "Training loss" ,"%.4f" % epoch_loss, "Time taken:","%.4f" % float(time.time() - time1))

    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        x_input = x_test[I_permutation[i:i + batch_size]]
        y_input = y_test[I_permutation[i:i + batch_size]]

        data = Variable(torch.FloatTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(data, target)

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

    print("  ", "Test Accuracy:" ,"%.2f" % (epoch_acc * 100.0),"Test Loss" , "Time taken: ","%.4f" % epoch_loss)

torch.save(model, 'BOW_b.model')
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('data_b.npy', data)


'''
Just like before, we need to try a few different hyperparameter settings and document the results.

Against the intuition laid out in the beginning of this section, this model actually seems to perform worse on average than the one in part a. This seems to achieve anywhere between 81-87%.

Let’s take a look at what’s happening. In part 1a, test accuracy typically seems to achieve its max after the 3rd epoch and begins to decrease with more training while the training accuracy continues to increase well into 90+%. This is a sure sign of overfitting. The training accuracy for part 1b stops much earlier (around 86-88%) and doesn’t seem to improve much more.

Nearly 95% of the weights belong to the embedding layer in part 1a. We’re training significantly less in part 1b and can’t actually fine-tune the word embeddings at all. Using only 300 hidden weights for part 1b results in very little overfitting while still achieving decent accuracy.
'''

##############
