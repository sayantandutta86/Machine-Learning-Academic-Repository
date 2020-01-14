# Author: Sayantan Dutta
# Program:  train_language_model.py
# Assignment 5
# Part- 3a

import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable
from RNN_language_model import RNN_language_model

import time
import io


#Hyper-parameters

resume_training = True

arg_vocab_size = 8000
arg_hidden_units = 500
arg_opt = 'adam'
arg_LR = 0.001
arg_batch_size = 200
arg_no_of_epochs = 75
arg_sequence_length_train = 50
arg_sequence_length_test = 100


imdb_dictionary = np.load('../preprocessed_data/imdb_dictionary.npy')
vocab_size = arg_vocab_size

x_train = []
with io.open('../preprocessed_data/imdb_train.txt','r',encoding='utf-8') as f:
    lines = f.readlines()
for line in lines:
    line = line.strip()
    line = line.split(' ')
    line = np.asarray(line,dtype=np.int)

    line[line>vocab_size] = 0

    x_train.append(line)


x_test = []
with io.open('../preprocessed_data/imdb_test.txt','r',encoding='utf-8') as f:
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

if resume_training:
    print('Resuming training from last saved checkpoint...')
    model = torch.load('temp.model')
    [train_loss, train_accu, test_accu, initial_epoch] = np.load('data.npy')
    [train_loss, train_accu, test_accu] = [list(train_loss), list(train_accu), list(test_accu)]
    optimizer = torch.load('temp.state')
    print('Initial Epoch.. ', initial_epoch)
else:
    print('Start training from the beginning...')
    model = RNN_language_model(vocab_size,arg_hidden_units)
    initial_epoch = 0
    train_loss = []
    train_accu = []
    test_accu = []
    if (arg_opt == 'adam'):
        optimizer = optim.Adam(model.parameters(), lr=arg_LR)
    elif (arg_opt == 'sgd'):
        optimizer = optim.SGD(model.parameters(), lr=arg_LR, momentum=0.9)

if torch.cuda.is_available():
    model = model.cuda()
else:
    print('No GPU...')
###############################################

# optimizer = 'sgd'
# LR = 0.01
#optimizer = arg_opt
#LR = arg_LR
# if(arg_opt=='adam'):
#     optimizer = optim.Adam(model.parameters(), lr=arg_LR)
# elif(arg_opt=='sgd'):
#     optimizer = optim.SGD(model.parameters(), lr=arg_LR, momentum=0.9)

batch_size = arg_batch_size
no_of_epochs = arg_no_of_epochs

L_Y_test = len(y_test)

model.train()



print('begin training...')
for epoch in range(initial_epoch, 75):

    if (epoch == 50):
        for param_group in optimizer.param_groups:
            param_group['lr'] = arg_LR / 10.0

    model.train()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(len(x_train))

    for i in range(0, len(x_train), batch_size):

        x_input2 = [x_train[j] for j in I_permutation[i:i + batch_size]]
        sequence_length = arg_sequence_length_train
        x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
        for j in range(batch_size):
            x = np.asarray(x_input2[j])
            sl = x.shape[0]
            if (sl < sequence_length):
                x_input[j, 0:sl] = x
            else:
                start_index = np.random.randint(sl - sequence_length + 1)
                x_input[j, :] = x[start_index:(start_index + sequence_length)]
        x_input = Variable(torch.LongTensor(x_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input)
        loss.backward()

        norm = nn.utils.clip_grad_norm_(model.parameters(), 2.0)

        optimizer.step()  # update gradients

        values, prediction = torch.max(pred, 1)
        prediction = prediction.cpu().data.numpy()
        accuracy = float(np.sum(prediction == x_input.cpu().data.numpy()[:, 1:])) / sequence_length
        epoch_acc += accuracy
        epoch_loss += loss.data.item()
        epoch_counter += batch_size

        if (i + batch_size) % 1000 == 0 and epoch == 0:
            print('Batch: ',i + batch_size, 'Accuracy/batch: ',accuracy / batch_size, 'Train Loss: ',loss.data.item(), 'Norm: ',norm, 'Time taken: ',"%.4f" % float(time.time() - time1))
    epoch_acc /= epoch_counter
    epoch_loss /= (epoch_counter / batch_size)

    train_loss.append(epoch_loss)
    train_accu.append(epoch_acc)

    print('\nEpoch:', epoch, 'Epoch train accuracy: ' ,"%.2f" % (epoch_acc * 100.0), 'Epoch train loss: ',"%.4f" % epoch_loss, 'Time taken: ',"%.4f" % float(time.time() - time1))

    ## test
    if ((epoch + 1) % 1 == 0):
        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):
            sequence_length = arg_sequence_length_test
            x_input2 = [x_test[j] for j in I_permutation[i:i + batch_size]]
            x_input = np.zeros((batch_size, sequence_length), dtype=np.int)
            for j in range(batch_size):
                x = np.asarray(x_input2[j])
                sl = x.shape[0]
                if (sl < sequence_length):
                    x_input[j, 0:sl] = x
                else:
                    start_index = np.random.randint(sl - sequence_length + 1)
                    x_input[j, :] = x[start_index:(start_index + sequence_length)]
            x_input = Variable(torch.LongTensor(x_input)).cuda()

            with torch.no_grad():
                pred = model(x_input, train=False)

            values, prediction = torch.max(pred, 1)
            prediction = prediction.cpu().data.numpy()
            accuracy = float(np.sum(prediction == x_input.cpu().data.numpy()[:, 1:])) / sequence_length
            epoch_acc += accuracy
            epoch_loss += loss.data.item()
            epoch_counter += batch_size
            # train_accu.append(accuracy)
            if (i + batch_size) % 1000 == 0 and epoch == 0:
                print(i + batch_size, accuracy / batch_size)
        epoch_acc /= epoch_counter
        epoch_loss /= (epoch_counter / batch_size)

        test_accu.append(epoch_acc)

        time2 = time.time()
        time_elapsed = time2 - time1

        print('\n', "  ",'Epoch test accuracy: ' , "%.2f" % (epoch_acc * 100.0), 'Epoch test loss: ', "%.4f" % epoch_loss, 'Time taken: ',"%.4f" % float(time.time() - time1))
    torch.cuda.empty_cache()

    if (((epoch + 1) % 2) == 0):
        torch.save(model, 'temp.model')
        torch.save(optimizer, 'temp.state')
        data = [train_loss, train_accu, test_accu, epoch]
        data = np.asarray(data)
        np.save('data.npy', data)
        print('Model saved at epoch' ,epoch, '...')
torch.save(model, 'language.model')

'''Gradient clipping is added in this training loop. Recurrent neural networks can sometimes experience extremely large gradients for a single batch which can cause them to be difficult to train without the gradient clipping.

This model takes much longer to train, about a day. The accuracy will be relatively low (which makes sense considering it’s trying to predict one of 8000 words) but this doesn’t actually tell you much. It’s better to go by the loss. Perplexity is typically used when comparing language models. More about how the cross entropy loss and perplexity are related can be read about here.

I trained multiple models with various results. The model here trained for 75 epochs with a sequence length of 50.'''