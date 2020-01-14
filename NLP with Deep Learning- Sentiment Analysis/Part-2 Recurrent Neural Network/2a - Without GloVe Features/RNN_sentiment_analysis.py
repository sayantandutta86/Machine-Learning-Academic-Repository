#Author: Sayantan Dutta
#Program:  RNN_sentiment_analysis.py
#part- 2a

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import time
import io

from RNN_model import RNN_model

#Hyper-parameters

arg_vocab_size = 8000
arg_hidden_units = 500
arg_opt = 'adam'
arg_LR = 0.001
arg_batch_size = 200
arg_no_of_epochs = 20
arg_sequence_length_train = 100
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
x_train = x_train[0:25000]
y_train = np.zeros((25000,))
y_train[0:12500] = 1

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

model = RNN_model(vocab_size,arg_hidden_units)
model.cuda()

###############################################

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
        y_input = y_train[I_permutation[i:i + batch_size]]

        data = Variable(torch.LongTensor(x_input)).cuda()
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(data, target, train=True)
        loss.backward()
		
'''In the RNN_model.py section, assuming the input x was batch_size by sequence_length. This means we’re always training on a fixed sequence length even though the reviews have varying length. We can do this by training on shorter subsequences of a fixed size. The variable x_input2 is a list with each element being a list of token ids for a single review. These lists within the list are different lengths. 

We create a new variable x_input which has our appropriate dimensions of batch_size by sequence_length which here I’ve specified as 100. We then loop through each review, get the total length of the review (stored in the variable sl), and check if it’s greater or less than our desired sequence_length of 100. If it’s too short, we just use the full review and the end is padded with 0s. If sl is larger than sequence_length (which it usually is), we randomly extract a subsequence of size sequence_length from the full review.

Notice during testing, we can choose a longer sequence_length such as 200 to get more context. However, due to the increased computation from the LSTM layer and the increased number of epochs, we may way to wrap the testing loop in the following to prevent it from happening everytime.'''		

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

    print(epoch, "Training accuracy: ","%.2f" % (epoch_acc * 100.0), "Training loss: ","%.4f" % epoch_loss, "Time taken: ","%.4f" % float(time.time() - time1))

    if (epoch + 1) % 5 == 0:
        print("Saving model...")
        torch.save(model, 'rnn.model')

    ### test
    if (epoch + 1) % 3 == 0:

        model.eval()

        epoch_acc = 0.0
        epoch_loss = 0.0

        epoch_counter = 0

        time1 = time.time()

        I_permutation = np.random.permutation(L_Y_test)

        for i in range(0, L_Y_test, batch_size):

            x_input2 = [x_train[j] for j in I_permutation[i:i + batch_size]]
            sequence_length = arg_sequence_length_test
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

        print("  ", "%.2f" % (epoch_acc * 100.0), "%.4f" % epoch_loss)



torch.save(model, 'rnn.model')
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('rnn_data.npy', data)

'''Choosing the sequence length is an important aspect of training RNNs and can have a very large impact on the results. By using long sequences, we provide the model with more information (it can learn longer dependencies) which seems like it’d always be a good thing. However, by using short sequences, we are providing less of an opportunity to overfit and artificially increasing the size of the dataset. There are more subsequences of length 50 than there are of length 100. Additionally, although we want our RNN to learn sequences, we don’t want it to overfit on where the phrases are within a sequence. By using subsequences, we are kind of training the model to be invariant about where it starts within the review which can be a good thing.

To make matters more complicated, we now have to decide how to test your sequences. If we train on too short of sequences, the internal state c may not have reached some level of steady state as it can grow unbounded if the network is never forced to forget any information. Therefore, when testing on long sequences, this internal state becomes huge as it never forgets things which results in an output distribution later layers have never encountered before in training and it becomes wildly inaccurate. This depends heavily on the application. For this particular dataset, I didn’t notice too many issues.

I’d suggest trying to train a model on short sequences (50 or less) as well as long sequences (250+) just to see the difference in its ability to generalize. The amount of GPU memory required to train an RNN is dependent on the sequence length. If we decide to train on long sequences, we may need to reduce your batch size.'''