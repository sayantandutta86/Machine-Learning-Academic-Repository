#Sayantan Dutta



# Code for iniitialization pytorch distributed

cmd = "/sbin/ifconfig"
out, err = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE,
    stderr=subprocess.PIPE).communicate()
ip = str(out).split("inet addr:")[1].split()[0]

name = MPI.Get_processor_name()
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
num_nodes = int(comm.Get_size())

ip = comm.gather(ip)

if rank != 0:
  ip = None

ip = comm.bcast(ip, root=0)

os.environ['MASTER_ADDR'] = ip[0]
os.environ['MASTER_PORT'] = '2222'

backend = 'mpi'
dist.init_process_group(backend, rank=rank, world_size=num_nodes)

dtype = torch.FloatTensor

## Main Program ##

import numpy as np
import pandas as pd
import os
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import argparse
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import torch.distributed as dist
import os
import subprocess
from mpi4py import MPI

data_path = '/u/training/tra272/scratch/hw4'
trained_model_path = '/u/training/tra272/scratch/hw4/cifar_sync/cifar_sync.t7'
save_model_path = '/u/training/tra272/scratch/hw4/cifar_sync'

resume_training = False

class BasicBlock(nn.Module):
    def __init__(self, channel_in, channel_out, stride, padding):
        super(BasicBlock, self).__init__()
        self.stride_flag = False
        self.conv3X3_block1 = nn.Conv2d(channel_in, channel_out, (3, 3), stride=stride, padding=padding)
        self.bnorm_block1 = nn.BatchNorm2d(channel_out)
        self.conv_down = nn.Conv2d(channel_in, channel_out, (1, 1), stride=stride)
        self.conv3X3_block2 = nn.Conv2d(channel_out, channel_out, (3, 3), stride=1, padding=padding)
        self.bnorm_block2 = nn.BatchNorm2d(channel_out)
        # Balance the downsample of the residual for stride>1
        if (stride > 1):
            self.stride_flag = True

    def forward(self, x):
        residual = x
        if (self.stride_flag):  # Balance the downsample of the residual for stride>1
            residual = self.conv_down(residual)
        x = self.conv3X3_block1(x)
        x = self.bnorm_block1(x)
        x = F.relu(x)
        x = self.conv3X3_block2(x)
        x = self.bnorm_block2(x)
        x += residual
        return x


# Implementing the Resnet architecture with Basic Block
class ResNet(nn.Module):
    def __init__(self, basic_block, output_units):  # Basic Block and Num of output classes
        super(ResNet, self).__init__()  # Specifying the model parameters
        # Input image is 3x32x32

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

        self.conv2_basic_layer_1 = basic_block(32, 32, 1, 1)
        self.conv2_basic_layer_2 = basic_block(32, 32, 1, 1)
        
        self.conv2_upsample = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv3_basic_layer_1 = basic_block(64, 64, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv3_basic_layer_2 = basic_block(64, 64, 1, 1)
        self.conv3_basic_layer_3 = basic_block(64, 64, 1, 1)
        self.conv3_basic_layer_4 = basic_block(64, 64, 1, 1)
        
        self.conv3_upsample = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv4_basic_layer_1 = basic_block(128, 128, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv4_basic_layer_2 = basic_block(128, 128, 1, 1)
        self.conv4_basic_layer_3 = basic_block(128, 128, 1, 1)
        self.conv4_basic_layer_4 = basic_block(128, 128, 1, 1)
        
        self.conv4_upsample = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv5_basic_layer_1 = basic_block(256, 256, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv5_basic_layer_2 = basic_block(256, 256, 1, 1)

        # MaxPool
        self.maxpool = nn.MaxPool2d(4, stride=4)

        # Fully connected layers with ReLU activation
        self.fc = nn.Linear(in_features=256, out_features=output_units)  # Fully connected layer at the end

        # Dropout
        self.dropout = nn.Dropout2d(p=0.2)

        # Batch Normalization
        self.bn_conv1 = nn.BatchNorm2d(32)

    def forward(self, x):  # Specifying the NN architecture
        x = F.relu(self.conv1(x))  # Convolution layers with relu activation
        x = self.bn_conv1(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = self.conv2_basic_layer_1(x)
        x = self.conv2_basic_layer_2(x)
        x = self.conv2_upsample(x)  # To balance the channel doubling change
        x = self.conv3_basic_layer_1(x)
        x = self.conv3_basic_layer_2(x)
        x = self.conv3_basic_layer_3(x)
        x = self.conv3_basic_layer_4(x)
        x = self.conv3_upsample(x)  # To balance the channel doubling change
        x = self.dropout(x)
        x = self.conv4_basic_layer_1(x)
        x = self.conv4_basic_layer_2(x)
        x = self.conv4_basic_layer_3(x)
        x = self.conv4_basic_layer_4(x)
        x = self.conv4_upsample(x)  # To balance the channel doubling change
        x = self.conv5_basic_layer_1(x)
        x = self.conv5_basic_layer_2(x)
        x = self.maxpool(x)  # MaxPooling layer
        x = x.view(x.size(0), -1)  # Flattening the conv2D output for dropout
        x = self.fc(x)  # Fully connected layer
        return x

intial_epoch = 0
num_epochs = 70
batch_size = 64


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

output_units = 100 #no. of classes in CIFAR-100
basic_block = BasicBlock

# Resume training from the last time?
if resume_training:
    # Load checkpoint
    print('Resuming training from last saved checkpoint ...')

    checkpoint = torch.load(trained_model_path)
    model = checkpoint['model']
    intial_epoch = checkpoint['epoch']
else:
    # start from the beginning
    print('\nTrain a new ResNet model ...')
    model = ResNet(basic_block, output_units)



gpu = torch.cuda.is_available()

#Initialize nodes

for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))
	
if gpu:
    model = model.cuda()

criterion = nn.CrossEntropyLoss()
LRate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=LRate, momentum=0.9)
scheduler = StepLR(optimizer, step_size = 2, gamma=0.95)

#Start training

start_time = time.time()
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(intial_epoch, num_epochs + intial_epoch):

    model.train()

    train_accuracy = []
    for i, input_batch in enumerate(trainloader, 0):
        input_image, labels = input_batch
        input_image, labels = Variable(input_image), Variable(labels)
        if gpu:
            input_image = input_image.cuda()
            labels = labels.cuda()
        optimizer.zero_grad()
        output = model(input_image)

        loss = criterion(output, labels)
        loss.backward()


        for param in model.parameters():

            if type(param.grad) == type(None):
                continue
            elif type(param.grad) != type(None):
                tensor0 = param.grad.data.cpu()
            dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
            tensor0 /= float(num_nodes)
            param.grad.data = tensor0.cuda()
            
        optimizer.step()
        prediction = output.data.max(1)[1]  # Label Prediction
        accuracy = (float(prediction.eq(labels.data).sum()) / float(
            batch_size)) * 100.0  # Computing the training accuracy
        train_accuracy.append(accuracy)


    accuracy_epoch = np.mean(train_accuracy)
    train_accuracy_list.append(accuracy_epoch)
    print('\nIn epoch ', epoch, ' the accuracy of the training set = ', accuracy_epoch)
    scheduler.step()

    #testing
    model.eval()

    test_accuracy = []

    for i, input_batch in enumerate(testloader, 0):
        input_image, labels = input_batch
        input_image, labels = Variable(input_image, volatile=True), Variable(labels, volatile=True)
        if gpu:
            input_image = input_image.cuda()
            labels = labels.cuda()
        output = model(input_image)
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(labels.data).sum()) / float(batch_size)) * 100.0
        test_accuracy.append(accuracy)
    accuracy_epoch = np.mean(test_accuracy)
    test_accuracy_list.append(accuracy_epoch)
    print('\nIn epoch ', epoch, ' the accuracy of the test set = ', accuracy_epoch)

    if epoch % 5 == 4:
        state = {'model': model, 'epoch': epoch}
        os.chdir(save_model_path)
        torch.save(state, 'cifar_sync.t7')
        print('\nModel saved... at epoch: ', epoch)

        plot_dict = {'training accuracy': train_accuracy_list, 'test accuracy': test_accuracy_list}
        x1 = pd.DataFrame.from_dict(plot_dict)
        x1.to_csv('cifar_sync_accuracy.csv')

end_time = time.time()
print('\nTime to train the model =', end_time - start_time)

#Save training and test accuracies for plotting
plot_dict = {'training accuracy': train_accuracy_list, 'test accuracy': test_accuracy_list }
x1 = pd.DataFrame.from_dict(plot_dict)
x1.to_csv('cifar_sync_accuracy.csv')