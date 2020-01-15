#Assignment 7
#Sayantan Dutta
# Part-1: Training a GAN on CIFAR10
#Train without generator

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import time
from DG_classes import discriminator, generator


from torch.autograd import Variable

import os
os.chdir('/u/training/tra272/scratch/hw7/hw7_2nd')

dataroot = "../data"
#ckptroot = "../model/"
ckptroot = '/u/training/tra272/scratch/hw7/hw7_2nd/model/'
option = 'option2'
lr = 0.0001
beta1 = 0
beta2 = 0.9
weight_decay = 1e-5
epochs1 = 60
epochs2 = 200
start_epoch = 0
batch_size_train = 128
batch_size_test = 128
n_z = 128
gen_train = 5
resume = False
cuda = True
num_classes = 10

class train_disc(object):
    """Trainer with generator."""

    def __init__(self, model, criterion, optimizer, trainloader, testloader, start_epoch, epochs, cuda, batch_size, learning_rate):
        """Trainer with generator BUilder."""
        super(train_disc, self).__init__()

        self.cuda = cuda
        self.model = model
        self.epochs = epochs
        self.criterion = criterion
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.testloader = testloader
        self.trainloader = trainloader
        self.start_epoch = start_epoch
        self.learning_rate = learning_rate

    def train(self):
        """Training discriminator without generator."""
        print("==> Start training ...")
        running_loss = 0.0

        for epoch in range(self.start_epoch, self.start_epoch + self.epochs):
            # learning rate decay
            if epoch == 40:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate / 10.0
            if epoch == 80:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.learning_rate / 100.0

            for batch_idx, (X_train_batch, Y_train_batch) in enumerate(self.trainloader):

                if Y_train_batch.shape[0] < self.batch_size:
                    continue

                if self.cuda:
                    X_train_batch = X_train_batch.cuda()
                    Y_train_batch = Y_train_batch.cuda()
                X_train_batch, Y_train_batch = Variable(
                    X_train_batch), Variable(Y_train_batch)
                _, output = self.model(X_train_batch)

                loss = self.criterion(output, Y_train_batch)
                self.optimizer.zero_grad()
                loss.backward()

                for group in self.optimizer.param_groups:
                    for p in group['params']:
                        state = self.optimizer.state[p]
                        if('step' in state and state['step'] >= 1024):
                            state['step'] = 1000
                self.optimizer.step()

                # print statistics
                running_loss += loss.data[0]

            # Normalizing the loss by the total number of train batches
            running_loss /= len(self.trainloader)

            # Calculate training/test set accuracy of the existing model
            train_accuracy = calculate_accuracy(
                self.model, self.trainloader, self.cuda)
            test_accuracy = calculate_accuracy(
                self.model, self.testloader, self.cuda)
            print("Iteration: {0} | Loss: {1} | Training accuracy: {2}% | Test accuracy: {3}%".format(
                epoch + 1, running_loss, train_accuracy, test_accuracy))

            if epoch % 5 == 0:
                print("==> Saving model at epoch: {}".format(epoch))
                if not os.path.isdir('model'):
                    os.mkdir('model')
                torch.save({'model':self.model,
                            'epoch': epoch
                            },'model/cifar10.pth')




def load_cifar10(root, batch_size_train, batch_size_test):

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
        transforms.ColorJitter(
            brightness=0.1 * torch.randn(1),
            contrast=0.1 * torch.randn(1),
            saturation=0.1 * torch.randn(1),
            hue=0.1 * torch.randn(1)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=root, train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size_train, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size_test, shuffle=False, num_workers=8)

    return trainloader, testloader


def calculate_accuracy(model, loader, cuda):

    correct = 0.
    total = 0.

    for data in loader:
        images, labels = data
        if cuda:
            images = Variable(images).cuda()
            labels = Variable(labels).cuda()
        _, outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum()

    return 100 * correct / total


def calc_gradient_penalty(netD, real_data, fake_data, batch_size, cuda):

    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(
        real_data.nelement() / batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    if cuda:
        alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    if cuda:
        interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                    grad_outputs=torch.ones(
                                        disc_interpolates.size()).cuda(),
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA

    return gradient_penalty

def plot(samples):

    fig = plt.figure(figsize=(10, 10))
    gs = gridspec.GridSpec(10, 10)
    gs.update(wspace=0.02, hspace=0.02)

    for i, sample in enumerate(samples):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(sample)
    return fig


#main program

# load cifar10 dataset
trainloader, testloader = load_cifar10(
    dataroot, batch_size_train, batch_size_test)

# Train the discriminator without the generator

print("Train the discriminator without the generator ...")
model = discriminator()

if resume:
    # Load checkpoint
    print('Resuming training from checkpoint ...')
    ckpt_pth = os.path.join(ckptroot, "cifar10.pth")
    checkpoint = torch.load(ckpt_pth)
    start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['model'].state_dict())

else:
    # start
    print("Train from the scratch ...")


if cuda:

    model = model.cuda()
    cudnn.benchmark = True
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),
                             lr=lr,
                             weight_decay=weight_decay)

# train
train_disc = train_disc(model, criterion, optimizer,
                       trainloader, testloader,
                       start_epoch, epochs1,
                       cuda, batch_size_train, lr)
train_disc.train()

