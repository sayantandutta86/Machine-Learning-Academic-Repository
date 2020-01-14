#Sayantan Dutta
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

resume_training = False

# resnet18_path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet/cifar100-finetune-code/resnet18/resnet18-5c106cde.pth'
# data_path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet'
# trained_model_path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet/cifar100-finetune-code/cifar_finetune.t7'
# save_model_path = '/content/drive/My Drive/Colab Notebooks/IE 534 Deep Learning/Resnet/cifar100-finetune-code'

resnet18_path = '/u/training/tra272/scratch/hw4/cifar_finetune_codes/resnet18-5c106cde.pth'
data_path = '/u/training/tra272/scratch/hw4'
trained_model_path = '/u/training/tra272/scratch/hw4/cifar_finetune_codes/cifar_finetune.t7'
save_model_path = '/u/training/tra272/scratch/hw4/cifar_finetune_codes'

os.chdir(save_model_path)

def tuned_pretrained_model(resnet18_path):
    model = torchvision.models.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    #model.load_state_dict(torch.utils.model_zoo.load_url(resnet18_path))
    model.load_state_dict(torch.load(resnet18_path))
    features = nn.Sequential(*list(model.children())[:-1])
    num_features = model.fc.in_features
    model.classifier = nn.Sequential(nn.Linear(num_features, 256),
                                    nn.ReLU(),
                                    nn.Dropout(0.2),
                                    nn.Linear(256, 100))
    return model

if resume_training:
    print('Resuming training from last saved checkpoint ...')
    checkpoint = torch.load(trained_model_path)
    model = checkpoint['model']
    intial_epoch = checkpoint['epoch']
else:
    model = tuned_pretrained_model(resnet18_path)

if torch.cuda.is_available():
    model = model.cuda()
    #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))

# Loss function, optimizer 
LRate=0.005
criterion = nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(model.parameters(), lr=LRate, weight_decay=1e-4)
optimizer = torch.optim.SGD(model.parameters(), lr=LRate, momentum=0.9)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.90)   
upscale = nn.Upsample(scale_factor=7, mode='bilinear')
intial_epoch = 0
num_epochs = 31
batch_size = 32

# Normalize training set together with augmentation
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Normalize test set same as training set without augmentation
transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

#Start training
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
start_time = time.time()
train_accuracy_list = []
test_accuracy_list = []
for epoch in range(intial_epoch, num_epochs+intial_epoch):

    model.train()
    
    train_accuracy = []
    for i, input_batch in enumerate(trainloader, 0):
        input_image, labels = input_batch
        input_image = upscale(input_image)
        input_image, labels = Variable(input_image).to(device), Variable(labels).to(device)
        optimizer.zero_grad() #Zero the gradients at each epoch
        output = model(input_image)#Forward propagation
        #Negative Log Likelihood Objective function
        loss = criterion(output, labels)
        loss.backward() #Backpropagation
        optimizer.step() #Updating the parameters using ADAM optimizer
        prediction = output.data.max(1)[1] #Label Prediction
        accuracy = (float(prediction.eq(labels.data).sum())/float(batch_size))*100.0 #Computing the training accuracy
        train_accuracy.append(accuracy)
    accuracy_epoch = np.mean(train_accuracy)
    train_accuracy_list.append(accuracy_epoch)
    print('\nIn epoch ', epoch,' the accuracy of the training set = ', accuracy_epoch)
    scheduler.step()
	
    with torch.no_grad():

        model.eval()

        test_accuracy = []

        for i, input_batch in enumerate(testloader, 0):
            input_image, labels = input_batch
            input_image = upscale(input_image)
            input_image, labels = Variable(input_image).to(device), Variable(labels).to(device)
            output = model(input_image)  #Forward propagation
            prediction = output.data.max(1)[1] #Label Prediction
            accuracy = (float(prediction.eq(labels.data).sum())/float(batch_size))*100.0
            test_accuracy.append(accuracy)
        accuracy_epoch = np.mean(test_accuracy)
        test_accuracy_list.append(accuracy_epoch)
        print('\nIn epoch ', epoch,' the accuracy of the test set = ', accuracy_epoch)

    if epoch%5==4:
        state = {'model': model, 'epoch': epoch}
        os.chdir(save_model_path)
        torch.save(state, 'cifar_finetune.t7')
        print('\nModel saved... at epoch: ', epoch)

end_time = time.time()
print('\nTime to train the model =', end_time - start_time)

plot_dict = {'training accuracy': train_accuracy_list, 'test accuracy': test_accuracy_list }
x1 = pd.DataFrame.from_dict(plot_dict)
x1.to_csv('cifar_finetune_accuracy.csv')

























