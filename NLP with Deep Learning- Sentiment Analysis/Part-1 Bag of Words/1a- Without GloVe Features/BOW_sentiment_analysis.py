#Author: Sayantan Dutta
#Program:  BOW_sentiment_analysis.py
#part-1a

#Import all of the basic packages we need as well our BOW_model we defined previously.
import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable

import time
import io

from BOW_model import BOW_model

#Hyper-parameters
arg_vocab_size = 8000
arg_opt = 'adam'
arg_LR = 0.001
arg_batch_size = 200
arg_no_of_epochs = 6
arg_hidden_units = 500


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


'''
Here is where we can much more quickly load in our data one time in the exact format we need. We can also choose our dictionary size at this point. I suggest starting out with 8000 as it was shown earlier (in the preprocessing data section) how this greatly reduces the number of weights in the word embedding layer without ignoring too many unique tokens within the actual dataset. Note that each line within our imdb_train.txt file is a single review made up of token ids. We can convert any token id greater than the dictionary size to the unknown token ID 0. Remember also we actually had 75,000 total training files but only the first 25,000 are labeled (we will use the unlabeled data in part 3). The last three lines of code just grab the first 25,000 sequences and creates a vector for the labels (the first 125000 are labeled 1 for positive and the last 12500 are labeled 0 for negative).
'''

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

model = BOW_model(vocab_size,arg_hidden_units)
model.cuda()

'''
Here we actually define the model with no_of_hidden_units equal to 500. Note I added 1 to vocab_size. Remember we actually added 1 to all of the token ids so we could use id 0 for the unknown token. The code above kept tokens 1 through 8000 as well as 0 meaning the vocab_size is actually 8001.
'''
#Define the optimizer with some desired learning rate.
# optimizer = 'sgd'
# LR = 0.01
opt = arg_opt
LR = arg_LR
if(opt=='adam'):
    optimizer = optim.Adam(model.parameters(), lr=LR)
elif(opt=='sgd'):
    optimizer = optim.SGD(model.parameters(), lr=LR, momentum=0.9)
	
'''
Define some parameters such as the batch_size and no_of_epochs. Put the model in training mode (this allows batch normalization to work correctly as well as dropout). Create a few lists to store any variables of interest.
'''

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
        x_input = [x_train[j] for j in I_permutation[i:i + batch_size]]
        y_input = np.asarray([y_train[j] for j in I_permutation[i:i + batch_size]], dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        optimizer.zero_grad()
        loss, pred = model(x_input, target)
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

    print(epoch, "Training accuracy: ","%.2f" % (epoch_acc * 100.0), "Training loss: ","%.4f" % epoch_loss, "Time taken: ","%.4f" % float(time.time() - time1))

    # ## test
    model.eval()

    epoch_acc = 0.0
    epoch_loss = 0.0

    epoch_counter = 0

    time1 = time.time()

    I_permutation = np.random.permutation(L_Y_test)

    for i in range(0, L_Y_test, batch_size):
        x_input = [x_test[j] for j in I_permutation[i:i + batch_size]]
        y_input = np.asarray([y_test[j] for j in I_permutation[i:i + batch_size]], dtype=np.int)
        target = Variable(torch.FloatTensor(y_input)).cuda()

        with torch.no_grad():
            loss, pred = model(x_input, target)

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

    print("  ", "Test accuracy: ","%.2f" % (epoch_acc * 100.0), "Test loss: ","%.4f" % epoch_loss)

torch.save(model, 'BOW.model')
data = [train_loss, train_accu, test_accu]
data = np.asarray(data)
np.save('data.npy', data)

'''
There is nothing particularly unique about this training loop compared to what you’ve seen in the past and it should be pretty self explanatory. The only interesting part is the variable x_input we send to the model is not actually a torch tensor at this moment. It’s simply a list of lists containing the token ids. Remember that this is dealt with in the BOW_model.forward() function.

With the current hyperparameters I’ve provided, this model should achieve around 86%-88% on the test dataset. If we check the original paper that came out with this dataset back in 2011, you’ll see this model performs just as well as nearly all of the techniques shown on page 7.

This should only take about 15-30 minutes to train. Report results for this model as well as at least two other trials (run more but only report on two interesting cases). Try to get a model that overfits (high accuracy on training data, lower on testing data) as well as one that underfits (low accuracy on both). Overfitting is easily achieved by greatly increasing the no_of_hidden_units, removing dropout, adding a second or third hidden layer in BOW_model.py as well as a few other things. Underfitting can be easily done by using significantly less hidden units.

Other things to consider:

Early stopping - notice how sometimes the test performance will actually get worse the longer we train
Longer training - this might be necessary depending on the choice of hyperparameters
Learning rate schedule - after a certain number of iterations or once test performance stops increasing, try reducing the learning rate by some factor
Larger models may take up too much memory meaning we may need to reduce the batch size
Play with the dictionary size - see if there is any difference utilizing a large portion of the dictionary (like 100k) compared to very few words (500)
we could try removing stop words, nltk.corpus.stopwords.words('english') returns a list of stop words. we can find their corresponding index using the imdb_dictionary and convert any of these to 0 when loading the data into x_train (remember to add 1 to the index to compensate for the unknown token)
Try SGD optimizer instead of adam. This might take more epochs. Sometimes a well tuned SGD with momentum performs better
'''