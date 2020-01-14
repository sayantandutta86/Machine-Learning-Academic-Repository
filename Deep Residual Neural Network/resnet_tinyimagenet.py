# Resnet with tiny-imagenet

import numpy as np
import pandas as pd
import os
import pdb
import time
import torch
import torchvision
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.utils.model_zoo as model_zoo
import argparse
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet/' #Colab path
# checkpoint_path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet/resnet_cifar100.t7' #Colab path
path = '/u/training/tra272/scratch/hw4'  # Bluewater path
checkpoint_path = '/u/training/tra272/scratch/hw4/tinyimagenet_codes/resnet_tiny.t7'  # Bluewater path

resume_training = False


def plot_graphs(plot_dict):
    for name, values in plot_dict.items():
        plt.plot(range(len(values)), values, label=name)
    plt.title('Train/Test Acccuracy of ResNet model in CIFAR-100')
    plt.legend()
    plt.xlabel("Epochs")
    plt.ylabel('Accuracy')
    plt.savefig('Accuracy Plot.png')


# Basic Block architecture
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
        # Input image is 3x64x64

        # convolution layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv1_a = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

        self.conv2_basic_layer_1 = basic_block(64, 64, 1, 1)
        self.conv2_basic_layer_2 = basic_block(64, 64, 1, 1)
        # To balance the downsampling effect due to stride=2, upsampling
        self.conv2_up = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=1)
        self.conv2_a = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv3_basic_layer_1 = basic_block(128, 128, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv3_basic_layer_2 = basic_block(128, 128, 1, 1)
        self.conv3_basic_layer_3 = basic_block(128, 128, 1, 1)
        self.conv3_basic_layer_4 = basic_block(128, 128, 1, 1)
        # To balance the downsampling effect due to stride=2, upsampling
        self.conv3_up = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv4_basic_layer_1 = basic_block(256, 256, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv4_basic_layer_2 = basic_block(256, 256, 1, 1)
        self.conv4_basic_layer_3 = basic_block(256, 256, 1, 1)
        self.conv4_basic_layer_4 = basic_block(256, 256, 1, 1)
        # To balance the downsampling effect due to stride=2, upsampling
        self.conv4_up = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)

        # Size of the image halves for stride=2
        self.conv5_basic_layer_1 = basic_block(512, 512, 2, 1)
        # Only the first Conv. layer has stride = 2, rest all have stride = 1
        self.conv5_basic_layer_2 = basic_block(512, 512, 1, 1)

        # MaxPool
        self.maxpool = nn.MaxPool2d(4, stride=4)

        # Fully connected layers with ReLU activation
        self.fc1 = nn.Linear(in_features=2048, out_features=512)  # Fully connected layer 1
        self.fc2 = nn.Linear(in_features=512, out_features=output_units)  # Fully connected layer at the end

        # Dropout
        self.dropout = nn.Dropout2d(p=0.1)

        # Batch Normalization
        self.bn_conv1 = nn.BatchNorm2d(32)
        self.bn_conv2 = nn.BatchNorm2d(64)
        self.bn_conv3 = nn.BatchNorm2d(128)
        self.bn_conv4 = nn.BatchNorm2d(256)
        self.bn_conv5 = nn.BatchNorm2d(512)

    def forward(self, x):  # Specifying the NN architecture
        x = F.relu(self.conv1(x))  # Convolution layers with relu activation
        x = self.bn_conv1(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = F.relu(self.conv1_a(x))  # Convolution layers with relu activation
        x = self.bn_conv2(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = self.conv2_basic_layer_1(x)
        x = self.conv2_basic_layer_2(x)
        x = self.conv2_up(x)  # To balance the channel doubling change
        x = F.relu(self.conv2_a(x))
        x = self.bn_conv3(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = self.conv3_basic_layer_1(x)
        x = self.conv3_basic_layer_2(x)
        x = self.conv3_basic_layer_3(x)
        x = self.conv3_basic_layer_4(x)
        x = self.conv3_up(x)  # To balance the channel doubling change
        x = self.bn_conv4(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = self.conv4_basic_layer_1(x)
        x = self.conv4_basic_layer_2(x)
        x = self.conv4_basic_layer_3(x)
        x = self.conv4_basic_layer_4(x)
        x = F.relu(self.conv4_up(x))  # To balance the channel doubling change
        x = self.bn_conv5(x)  # Batch normalization
        x = self.dropout(x)  # Dropout
        x = self.conv5_basic_layer_1(x)
        x = self.conv5_basic_layer_2(x)
        #print('Training mode = ', training_mode)
        #print('x-size before maxpool ',x.size())	
        x = self.maxpool(x)  # MaxPooling layer
        #print('x-size after maxpool ',x.size())	
        x = x.view(x.size(0), -1)  # Flattening the conv2D output for dropout
        #print('x-size ',x.size())		
        #print('x-size after flatteing ',x.size(),'\n')	
        #pdb.set_trace()		
        x = F.relu(self.fc1(x))  # Fully connected layer 1
        x = self.fc2(x)
        #x = F.softmax(self.fc2(x), dim= -1) # Fully connected layer 1
        return x


intial_epoch = 0
num_epochs = 40
batch_size = 64

transform_train = transforms.Compose([
    transforms.RandomCrop(64, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dir = '/u/training/tra272/scratch/hw4/tiny-imagenet-200/train'
train_dataset = datasets.ImageFolder(train_dir, transform=transform_train)
# print(train_dataset.class_to_idx)
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True, num_workers=8)

val_dir = '/u/training/tra272/scratch/hw4/tiny-imagenet-200/val/images'
val_dataset = datasets.ImageFolder(val_dir, transform=transform_test)

# print(val_dataset.class_to_idx)
val_loader = torch.utils.data.DataLoader(val_dataset,
                                         batch_size=batch_size, shuffle=False, num_workers=8)
'''
for images, labels in train_loader:
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
for images, labels in val_loader:
    images = Variable(images).to(device)
    labels = Variable(labels).to(device)
'''

# Start training

output_units = 200  # no. of classes in CIFAR-100
basic_block = BasicBlock

# Resume training from the last time?
if resume_training:
    # Load checkpoint
    print('Resuming training from last saved checkpoint ...')
    assert os.path.isdir(
        path), 'Error: no checkpoint directory found!'
    checkpoint = torch.load(checkpoint_path)
    model = checkpoint['model']
    intial_epoch = checkpoint['epoch']
    print('\nModel loaded..  training to resume from last saved checkpoint ...') 	
else:
    # start from the beginning
    print('\nTrain a new ResNet model ...')
    model = ResNet(basic_block, output_units)

model = model.to(device)

if torch.cuda.is_available():
    model = model.cuda()
    model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
LRate = 0.005
optimizer = torch.optim.SGD(model.parameters(), lr=LRate, momentum=0.9)
scheduler = StepLR(optimizer, step_size=2, gamma=0.95)



start_time = time.time()
train_accuracy_list = []
test_accuracy_list = []

for epoch in range(intial_epoch, num_epochs + intial_epoch):
    print('\n','Epoch ', epoch, ' begins...')  

    model.train()

    train_accuracy = []
    for i, input_batch in enumerate(train_loader, 0):
        input_image, labels = input_batch
        input_image, labels = Variable(input_image).to(device), Variable(labels).to(device)
        #print('Length of input image batch: ',len(input_image))
        #pdb.set_trace()
        optimizer.zero_grad()  # Zero the gradients at each epoch
        output = model(input_image)  # Forward propagation
        # Negative Log Likelihood Objective function
        loss = criterion(output, labels)
        loss.backward()  # Backpropagation
        optimizer.step()  # Updating the parameters using ADAM optimizer
        prediction = output.data.max(1)[1]  # Label Prediction
        accuracy = (float(prediction.eq(labels.data).sum()) / float(batch_size)) * 100.0  # Computing the training accuracy
        train_accuracy.append(accuracy)
    accuracy_epoch = np.mean(train_accuracy)
    train_accuracy_list.append(accuracy_epoch)
    scheduler.step()
    print('\nIn epoch ', epoch+1, ' the accuracy of the training set = ', accuracy_epoch)

    with torch.no_grad():
        model.eval()
        training_mode = False   		

        test_accuracy = []

        for i, input_batch in enumerate(val_loader, 0):
            input_image, labels = input_batch
            input_image, labels = Variable(input_image).to(device), Variable(labels).to(device)
            output = model(input_image)  # Forward propagation
            prediction = output.data.max(1)[1]  # Label Prediction
            accuracy = (float(prediction.eq(labels.data).sum()) / float(batch_size)) * 100.0
            test_accuracy.append(accuracy)
        accuracy_epoch = np.mean(test_accuracy)
        test_accuracy_list.append(accuracy_epoch)
        print('\nIn epoch ', epoch+1, ' the accuracy of the test set = ', accuracy_epoch)

    if epoch % 5 == 4:
        state = {'model': model, 'epoch': epoch}
        os.chdir(path)
        torch.save(state, 'resnet_tiny.t7')
        print('\nModel saved... at epoch: ', epoch)

end_time = time.time()
print('\nTime to train the model =', end_time - start_time)

plot_dict = {'training accuracy': train_accuracy_list, 'test accuracy': test_accuracy_list }
x1 = pd.DataFrame.from_dict(plot_dict)
x1.to_csv('tiny_accuracy.csv')









