# Human Action Recognition.
#Part 2 - Sequence Model - Train
#Sayantan Dutta

'''Although the single frame model achieves around 73%-75% classification accuracy for a single frame, it above should achieve between 77%-79% after combining the single frame predictions over the whole video. As mentioned above, simply averaging these predictions is a very naive way of classifying sequences. This is similar to the Bag of Words model from the NLP assignment. There are many ways to combine this information in a more intelligent way.

There are many ways to utilize the temporal information. All of the papers in the introduction essentially explore these different techniques. Part 2 of the assignment will do this by using 3D convolutions. 3D convolutions are conceptually the exact same as 2D convolutions except now they also operate over the temporal dimension (sliding window over the frames as well as the image).

The model already overfits on single frames alone. If you were to train a 3D convolutional network from scratch on UCF-101, it severely overfits and has extremely low performance. The Kinetics dataset is a much larger action recognition dataset released more recently than UCF-101. The link above goes to the Kinetics-600 dataset (500,000 videos of 600 various actions). We will use a a 3D ResNet-50 model pretrained on the Kinetics-400 dataset (300,000 videos of 400 various actions) from here. This pretrained model is located in the class directory /projects/training/bayw/hdf5/UCF-101-hdf5 on BlueWaters.'''


import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import resnet_3d

import h5py
import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 101
# batch_size = 32
batch_size = 16
lr = 0.0001
num_of_epochs = 10


data_directory = '/projects/training/bayw/hdf5/'
class_list, train, test = getUCF101(base_directory = data_directory)

model =  resnet_3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)
pretrained = torch.load('resnet-50-kinetics.pth')
keys = [k for k,v in pretrained['state_dict'].items()]
pretrained_state_dict = {k[7:]: v.cpu() for k, v in pretrained['state_dict'].items()}
model.load_state_dict(pretrained_state_dict)
model.fc = nn.Linear(model.fc.weight.shape[1],NUM_CLASSES)

'''When creating the model object, we are defining the sample_duration=16. That is, we are going to be training on subsequences of 16 frames. As the input passes through the network, there will be max pooling operations over the spatial and temporal dimensions as well as strided 3D convolutions which will reduce the spatial and temporal dimensions by a factor of 2. By the time the final layer is processed, the temporal dimension has collapsed meaning you have a single vector output for all 16 frames.

The state dictionary for the pretrained model is loaded into pretrained. The names of the of layers are used to place the weights into the model we defined. Redefine the fully connected layer once again such that it has the appropriate number of outputs.'''

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
for param in model.layer4[0].parameters():
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
for param in model.layer4[0].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)


model.cuda()

optimizer = optim.Adam(params,lr=lr)

criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8,maxtasksperchild=200)

'''This is very similar to the single frame model as in we will only fine-tune the output layer and the last residual block. Note that model.layer4 actually has three residual blocks. However, BlueWaters does not have a large enough GPU to handle the blocks model.layer4[1] and model.layer4[2]. The largest attainable batch size with these layers included is 1 which is not enough to train the model due to batch normalization. Instead, we will simply ignore these last two blocks and directly send the output of model.layer4[0] into the fully connected layer. How this is done will be shown in the training loop.'''

for epoch in range(0,num_of_epochs):

    # TRAIN #
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()
    for i in range(0, len(train[0])-batch_size,batch_size):

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x),requires_grad=False).cuda().contiguous()

        y = train[1][random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4[0](h)

        h = model.avgpool(h)

        h = h.view(h.size(0), -1)
        output = model.fc(h)

        # output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()
        
        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        if(epoch==0):
            print(i,accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch,time.time()-start_time)

    # EVAL #
    model.eval()
    test_accu = []
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()
    for i in range(0,len(test[0])-batch_size,batch_size):
        augment = False
        video_list = [(test[0][k],augment) 
                        for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)
		'''Here we are using the function loadSequence() instead of loadFrame(). This will grab a subsequence of 16 frames from the video for training/testing. The input x will now be size [batch_size,3,16,224,224].'''

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this batch
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()

        y = test[1][random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        # with torch.no_grad():
        #     output = model(x)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            # h = model.layer4[1](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)
			
			'''The above code is how to manage avoiding the use of model.layer4[1] and model.layer4[2] to save GPU memory. Instead of calling the forward() function of model, we can manually apply each layer. The first portion of the network can be run under with torch.no_grad(): since these weights will not be trained.

			The last two residual blocks in model.layer4 are already trained and provide valuable features. If it was possible, they should both be used. However, the model still performs very well even without these two layers. During the training loop, the training accuracy will once again max out at around 100% and the testing accuracy should be around 81%-83%.'''

        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        test_accu.append(accuracy)
        accuracy_test = np.mean(test_accu)

    print('Testing',accuracy_test,time.time()-t1)

torch.save(model,'3d_resnet.model')
pool_threads.close()
pool_threads.terminate()

'''There are really only two differences when compared to the single frame training script.'''