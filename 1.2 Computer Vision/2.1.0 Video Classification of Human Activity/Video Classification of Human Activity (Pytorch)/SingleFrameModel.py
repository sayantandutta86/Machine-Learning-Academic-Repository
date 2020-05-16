# Human Action Recognition.
#Part 1 - Single Frame Model
#Sayantan Dutta

'''This first portion of the assignment uses a pretrained ResNet-50 model pretrained on ImageNet. Although the dataset has a large number of frames (13000 videos * number of frames per video), the frames are correlated with eachother meaning there isn’t a whole lot of variety. Also, to keep the sequences relatively short (~2-10 seconds), some of the original videos were split up into 5-6 shorter videos meaning there is even less variety. Single frames alone can still provide a significant amount of information about the action being performed (consider the classes “Skiing” versus “Baseball Pitch”). Training a CNN from scratch significantly overfits. However, the features from a CNN pretrained on ImageNet (over 1 million images of 1000 classes) can be very useful even in video based problem like action recognition. A single frame model performs surprisingly well. This doesn’t necessarily mean solving the task of learning from images inherently solves all video related tasks. It’s more likely that with the problem of human action recognition, the spatial information is more important than the temporal information.'''

import numpy as np
import os
import sys
import time

import h5py
import cv2

import torch
import torchvision
import torch.nn as nn
import torch.optim as optim

from helperFunctions import getUCF101
from helperFunctions import loadFrame
from torch.autograd import Variable
from multiprocessing import Pool



IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 100
lr = 0.0001
num_of_epochs = 10
data_directory = '/projects/training/bayw/hdf5/'

class_list, train, test = getUCF101(base_directory=data_directory)

'''Import the necessary python modules, set a few basic hyperparameters, and load the dataset variables class_list, train, and test.'''

model = torchvision.models.resnet50(pretrained=True)

# overwrite the last fully connected layer such that it has the number of
# outputs equal to the number of classes
model.fc = nn.Linear(2048, NUM_CLASSES)

for param in model.parameters():
    param.requires_grad_(False)

# for param in model.conv1.parameters():
#     param.requires_grad_(True)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
for param in model.layer4[2].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []
# for param in model.conv1.parameters():
#     params.append(param)
# for param in model.bn1.parameters():
#     params.append(param)
# for param in model.layer1.parameters():
#     params.append(param)
# for param in model.layer2.parameters():
#     params.append(param)
# for param in model.layer3.parameters():
#     params.append(param)
for param in model.layer4[2].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)

model.cuda()
optimizer = optim.Adam(params, lr=lr)
criterion = nn.CrossEntropyLoss()

'''The module torchvision comes with a pretrained ResNet-50 model. Overwrite the last fully connected layer such that it has the number of outputs equal to the number of classes. As mentioned before, the dataset is not large enough to warrant training a full ResNet-50 model. We will just fine-tune the output layer and the last residual block. model.layer4 contains three residual blocks implying model.layer4[2] is the last of these three residual blocks. Fine-tuning only the top layers also reduces the amount of GPU memory meaning a higher batch size can be used and the model can be trained more quickly with less overfitting.'''

# leverage multiple CPU cores using a Pool() object
pool_threads = Pool(8, maxtasksperchild=200)

'''As mentioned previously, this code does not use a PyTorch dataset/dataloader. However, we can still leverage multiple CPU cores using a Pool() object with our dataloading function loadFrame(). How this is done will become apparent in the training loop.'''

for epoch in range(0, num_of_epochs):
    
    # TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()

    for i in range(0, len(train[0]) - batch_size, batch_size):
        augment = True
        video_list = [(train[0][k], augment)
                      for k in random_indices[i:(batch_size + i)]]
        data = pool_threads.map(loadFrame, video_list)

        next_batch = 0
        for video in data:
            # there was an exception, skip this
            if video.size == 0:
                next_batch = 1
        if next_batch == 1:
            continue

        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()
		
		'''This portion of the code grabs a batch of data by first creating a list of tuples with the tuples being video filepaths (train[0][k]) and data augmentation (augment=True). The pool_threads.map() function takes as input a function (loadFrame()) and a list of arguments (video_list) for this function. The function is called for every tuple in the list. The Pool() object was created with the ability to run execute these commands on eight separate CPU cores. The final output is going to be a list of data frames (size [3,224,224]). Some of the frames from the hdf5 files can be corrupt which causes an exception every so often. The second part of the code is simply a hack way of skipping over a particular iteration if any of the function calls fails. Lastly, the list of data frames is converted to a PyTorch variable of size [batch_size,3,224,224] and moved to the GPU so it can be input to the ResNet-50 model.'''
		
        y = train[1][random_indices[i:(batch_size + i)]]
        y = torch.from_numpy(y).cuda()
        output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(y.data).sum()) /
                    float(batch_size)) * 100.0

        if epoch == 0:
            print(">>>    Batch: {} | Accuracy: {}".format(i, accuracy))
        train_accu.append(accuracy)

    accuracy_epoch = np.mean(train_accu)
    print("Training Epoch: {} , Accuracy : {} , Elapsed time: {}".format(
        epoch, accuracy_epoch, time.time() - start_time))

    torch.save(model, 'single_frame.model')

    # data augmentation
    augment = True
    video_list = [(train[0][k], augment)
                  for k in random_indices[i:(batch_size + i)]]
    data = pool_threads.map(loadFrame, video_list)

    next_batch = 0
    for video in data:
        # there was an exception, skip this
        if video.size == 0:
            next_batch = 1
    if next_batch == 1:
        continue

    x = np.asarray(data, dtype=np.float32)
    x = Variable(torch.FloatTensor(x)).cuda().contiguous()

    
    # EVAL
    model.eval()
    test_accu = []
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0, len(test[0]) - batch_size, batch_size):
        augment = False
        video_list = [(test[0][k], augment)
                      for k in random_indices[i:(batch_size + i)]]
        data = pool_threads.map(loadFrame, video_list)

        next_batch = 0
        for video in data:
            # there was an exception, skip this batch
            if video.size == 0:
                next_batch = 1
        if next_batch == 1:
            continue

        x = np.asarray(data, dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()
        y = test[1][random_indices[i:(batch_size + i)]]
        y = torch.from_numpy(y).cuda()
        output = model(x)

        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(y.data).sum()) /
                    float(batch_size)) * 100.0
        test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)

    print("Validation Accuracy: {} , Elapsed time: {}".format(
        accuracy_test, time.time() - t1))
		
	'''After ~10 epochs, the model will be achieving almost 100% on the training data. To back up the statement before about the frames within a video being highly correlated, after 10 epochs, this means the model has only seen 10 random frames from each video. However, it can correctly classify any other random unseen frame in the training data with near perfect accuracy while having only around 73%-75% accuracy on the test data.'''
		
pool_threads.close()
pool_threads.terminate()


# TEST

'''This test accuracy is only for a single frame. At true test time, it would make sense to average out the prediction over every frame within a video. After training, let’s create a loop to calculate predictions for the entire test dataset. This can be done in a separate file or after the training loop.'''

model = torch.load('single_frame.model')
model.cuda()

# save predictions directory
prediction_directory = 'UCF-101-predictions/single_frame/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory + label + '/'):
        os.makedirs(prediction_directory + label + '/')


'''The above creates a directory structure laid out in the exact same way the dataset is structured. Predictions for each video can be saved as numpy arrays stored in hdf5 format and the train and test list can be used to load these in at a future time.'''

acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0
confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406], np.float32)
std = np.asarray([0.229, 0.224, 0.225], np.float32)
model.eval()


'''Create some variables to track the (top 1,top 5,top 10) accuracy. The top_10 accuracy simply signifies how often the correct class is any of the top 10 predicted classes. A confusion matrix is a square matrix of size [NUM_CLASSES,NUM_CLASSES]. The confusion matrix is used to show how often each particular class is classified by the model as other classes. That is, for each video, the value of confusion_matrix[actual_class,predicted_class] is increased by 1. After going through the entire dataset, if you divide each row by its row sum, it will provide a percentage breakdown of this class confusion. A perfect classifier would have 100% for every diagonal element and 0% for all of the off diagonal elements.'''

all_prediction = np.zeros((len(test[0]), NUM_CLASSES), dtype=np.float32)

for i in range(len(test[0])):

    t1 = time.time()
    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi', '.hdf5')
    filename = filename.replace('UCF-101', 'UCF-101-hdf5')

    h = h5py.File(filename, 'r')
    nFrames = len(h['video'])

    data = np.zeros((nFrames, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)

    for j in range(nFrames):
        frame = h['video'][j]
        frame = frame.astype(np.float32)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame / 255.0
        frame = (frame - mean) / std
        frame = frame.transpose(2, 0, 1)
        data[j, :, :, :] = frame
    h.close()

    prediction = np.zeros((nFrames, NUM_CLASSES), dtype=np.float32)

    loop_i = list(range(0, nFrames, 200))
    loop_i.append(nFrames)

    for j in range(len(loop_i) - 1):
        data_batch = data[loop_i[j]:loop_i[j + 1]]

        with torch.no_grad():
            x = np.asarray(data_batch, dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()
            output = model(x)

        prediction[loop_i[j]:loop_i[j + 1]] = output.cpu().numpy()

		'''The numpy array predictions will hold the probabilities for each class for each frame. It is possible to loop through the frames one at a time and grab the output from the model. However, this would be very slow and not utilize any batch processing. You could process the full video at once (with the sequence_length basically being the batch_size), but some of the sequences are too long to fit in memory. This loop breaks the video into subsequences of length 200 (which does fit on the GPU), performs batch processing, sets the prediction variable equal to the output for the corresponding frames and continues until the full video has been passed through the model.'''


    # saves the `prediction` array in hdf5 format
    filename = filename.replace(
        data_directory + 'UCF-101-hdf5/', prediction_directory)
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as h:
            h.create_dataset('predictions', data=prediction)

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j]) / np.sum(np.exp(prediction[j]))

    prediction = np.sum(np.log(prediction), axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    all_prediction[index, :] = prediction / nFrames

    label = test[1][index]
    confusion_matrix[label, argsort_pred[0]] += 1
    if label == argsort_pred[0]:
        acc_top1 += 1.0
    if np.any(argsort_pred[0:5] == label):
        acc_top5 += 1.0
    if np.any(argsort_pred[:] == label):
        acc_top10 += 1.0

    print('>>> i:%d nFrames:%d t:%f (%f, %f, %f)'
          % (i, nFrames, time.time() - t1, acc_top1 / (i + 1), acc_top5 / (i + 1), acc_top10 / (i + 1)))


	'''The final part of the loop first saves the prediction array in hdf5 format. The softmax operation is used on the output providing class probabilities for each frame. The line prediction = np.sum(np.log(prediction),axis=0) is a naive way of calculating log(P(Y|X)) and choosing the most likely class by assuming each frame is independent of the other frames (although they’re not independent). prediction can also be used to get the 10 most likely classes to calculate the different accuracies.'''	

number_of_examples = np.sum(confusion_matrix, axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i, :] = confusion_matrix[i, :] / \
        np.sum(confusion_matrix[i, :])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i], sorted_results[i], number_of_examples[indices[i]])

np.save('single_frame_confusion_matrix.npy', confusion_matrix)
np.save('single_frame_prediction_matrix.npy', all_prediction)

'''After the loop, the confusion matrix can be converted to confusion probabilities. The diagonal elements will say how often a particular class is correctly identified. The code above sorts these values and prints them all out. The saved predictions and confusion_matrix will be used later in the assignment.'''