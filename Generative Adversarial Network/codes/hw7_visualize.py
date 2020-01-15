#Assignment 7
#Sayantan Dutta
#Part 2 Visualize

import torch
import torchvision
import torch.utils.data
import torchvision.transforms as transforms

import os
import torch
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from torch.autograd import Variable


import torch.nn as nn
import torch.nn.functional as F

from DG_classes import discriminator, generator
num_classes=10
section = 'option3'
dataroot = "../data"
modelroot = "./"
batch_size_test = 128
resume = False
cuda = True

class discriminatorBottom(nn.Module):
    def __init__(self, extract_features):
        super(discriminatorBottom, self).__init__()

        self.conv1 = nn.Conv2d(3, 196, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.conv5 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv6 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(196, 196, kernel_size=3, stride=2, padding=1)
        self.layer_norm1 = nn.LayerNorm((196, 32, 32))
        self.layer_norm2 = nn.LayerNorm((196, 16, 16))
        self.layer_norm3 = nn.LayerNorm((196, 16, 16))
        self.layer_norm4 = nn.LayerNorm((196, 8, 8))
        self.layer_norm5 = nn.LayerNorm((196, 8, 8))
        self.layer_norm6 = nn.LayerNorm((196, 8, 8))
        self.layer_norm7 = nn.LayerNorm((196, 8, 8))
        self.layer_norm8 = nn.LayerNorm((196, 4, 4))
        self.fc1 = nn.Linear(196, 1)
        self.fc10 = nn.Linear(196, num_classes)


    def forward(self, x):

        x = F.leaky_relu(self.layer_norm1(self.conv1(x)))
        x = F.leaky_relu(self.layer_norm2(self.conv2(x)))
        if  extract_features==2:
            x = F.max_pool2d(x, kernel_size=16, padding=0, stride=16)
            x = x.view(x.size(0), -1)
            #out1 = self.fc1(x)
            print('At feature layer 2...')
            return x
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)))
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)))
        print('extract features = ', extract_features)
        if  extract_features==4:
            x = F.max_pool2d(x, kernel_size=8, padding=0, stride=8)
            x = x.view(x.size(0), -1)
            #out1 = self.fc1(x)
            print('At feature layer 4...')
            return x

        x = F.leaky_relu(self.layer_norm5(self.conv5(x)))
        x = F.leaky_relu(self.layer_norm6(self.conv6(x)))
        x = F.leaky_relu(self.layer_norm7(self.conv7(x)))
        x = F.leaky_relu(self.layer_norm8(self.conv8(x)))

        x = F.max_pool2d(x, kernel_size=4, padding=0, stride=4)

        x = x.view(x.size(0), -1)
        #out1 = self.fc1(x)
        out10 = self.fc10(x)
        return out10


#section-1
def real_images_perturb(testloader, modelroot, batch_size, cuda):

    testloader = enumerate(testloader)


    checkpoint = torch.load('/u/training/tra272/scratch/hw7/hw7_2nd/model/cifar10.pth')
    model = checkpoint['model']
    if cuda:
        model.cuda()
    model.eval()

    batch_idx, (X_batch, Y_batch) = testloader.__next__()
    X_batch = Variable(X_batch, requires_grad=True).cuda()
    Y_batch_alternate = (Y_batch + 1) % 10
    Y_batch_alternate = Variable(Y_batch_alternate).cuda()
    Y_batch = Variable(Y_batch).cuda()


    samples = X_batch.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    if not os.path.isdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization'):
        os.mkdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization')
    plt.savefig('/u/training/tra272/scratch/hw7/hw7_2nd/visualization/real_images.png', bbox_inches='tight')
    plt.close(fig)

    _, output = model(X_batch)
    # first column has actual prob.
    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y_batch.data).sum()) /
                float(batch_size)) * 100.0
    print("Original Image | Accuracy: {}%".format(accuracy))


    criterion = nn.CrossEntropyLoss(reduce=False)
    loss = criterion(output, Y_batch_alternate)

    gradients = torch.autograd.grad(outputs=loss, inputs=X_batch,
                                    grad_outputs=torch.ones(
                                        loss.size()).cuda(),
                                    create_graph=True, retain_graph=False, only_inputs=True)[0]

    # save gradient
    gradient_image = gradients.data.cpu().numpy()
    gradient_image = (gradient_image - np.min(gradient_image)) / \
        (np.max(gradient_image) - np.min(gradient_image))
    gradient_image = gradient_image.transpose(0, 2, 3, 1)
    fig = plot(gradient_image[0:100])
    plt.savefig('/u/training/tra272/scratch/hw7/hw7_2nd/visualization/gradient_image.png', bbox_inches='tight')
    plt.close(fig)


    gradients[gradients > 0.0] = 1.0
    gradients[gradients < 0.0] = -1.0

    gain = 8.0
    X_batch_modified = X_batch - gain * 0.007843137 * gradients
    X_batch_modified[X_batch_modified > 1.0] = 1.0
    X_batch_modified[X_batch_modified < -1.0] = -1.0

    # evaluate new fake images
    _, output = model(X_batch_modified)
    # first column has actual prob
    prediction = output.data.max(1)[1]
    accuracy = (float(prediction.eq(Y_batch.data).sum()) /
                float(batch_size)) * 100.0
    print("Jitter Input Image | Accuracy: {}%".format(accuracy))

    # save fake images
    samples = X_batch_modified.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    plt.savefig('/u/training/tra272/scratch/hw7/hw7_2nd/visualization/jittered_images.png', bbox_inches='tight')
    plt.close(fig)

#section-2
def max_class_function(testloader, model, cuda, fig_name):

    testloader = enumerate(testloader)

    if cuda:
        model.cuda()
    model.eval()
    batch_idx, (X_batch, Y_batch) = testloader.__next__()

    X = X_batch.mean(dim=0)
    X = X.repeat(10, 1, 1, 1)
    X = Variable(X, requires_grad=True).cuda()

    Y = torch.arange(10).type(torch.int64)
    Y = Variable(Y).cuda()

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        #_, _, output = model(X)
        _, output = model(X)

        loss = -output[torch.arange(10).type(torch.int64),
                       torch.arange(10).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(
                                            loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]

        # first column has actual prob.
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y.data).sum()) / float(10.0)) * 100.0
        print("Ieration: {} | Accuracy: {} | Loss: {}".format(
            i, accuracy, -loss.data[0]))

        X = X - lr * gradients.data - weight_decay * X.data * torch.abs(X.data)
        X[X > 1.0] = 1.0
        X[X < -1.0] = -1.0

    # save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples)
    if not os.path.isdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization'):
        os.mkdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization')
    figname2 = '/u/training/tra272/scratch/hw7/hw7_2nd/visualization/max_class' + fig_name + '.png'
    plt.savefig(figname2, bbox_inches='tight')
    plt.close(fig)
	#The example above changes each pixel by 10⁄255 based on the gradient sign.

#section-3
def max_features_function(testloader, model, batch_size, cuda, extract_features, fig_name):

    testloader = enumerate(testloader)



    if cuda:
        model.cuda()
    model.eval()
    batch_idx, (X_batch, Y_batch) = testloader.__next__()

    X = X_batch.mean(dim=0)
    X = X.repeat(batch_size, 1, 1, 1)
    X = Variable(X, requires_grad=True).cuda()

    Y = torch.arange(batch_size).type(torch.int64)
    Y = Variable(Y).cuda()

    lr = 0.1
    weight_decay = 0.001
    for i in range(200):
        # if train D without G
        # output, _, _ = model(X)
        # else if train D with G
        output = model(X)

        # loss = -output[torch.arange(batch_size).type(torch.int64),
        #                torch.arange(batch_size).type(torch.int64)]
        loss = -output[torch.arange(batch_size).type(torch.int64)]
        gradients = torch.autograd.grad(outputs=loss, inputs=X,
                                        grad_outputs=torch.ones(
                                            loss.size()).cuda(),
                                        create_graph=True, retain_graph=False, only_inputs=True)[0]

        # first column has actual prob
        prediction = output.data.max(1)[1]
        accuracy = (float(prediction.eq(Y.data).sum()) /
                    float(batch_size)) * 100.0
        print("Iteration: {} | Accuracy: {} | Loss: {}".format(
            i, accuracy, -loss.data[0]))

        X = X - lr * gradients.data - weight_decay * X.data * torch.abs(X.data)
        X[X > 1.0] = 1.0
        X[X < -1.0] = -1.0

    # save new images
    samples = X.data.cpu().numpy()
    samples += 1.0
    samples /= 2.0
    samples = samples.transpose(0, 2, 3, 1)

    fig = plot(samples[0:100])
    if not os.path.isdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization'):
        os.mkdir('/u/training/tra272/scratch/hw7/hw7_2nd/visualization')
    figname2 = '/u/training/tra272/scratch/hw7/hw7_2nd/visualization/max_features_' + str(fig_name) + '.png'
    plt.savefig(figname2, bbox_inches='tight')
    plt.close(fig)

def plot(samples):
    """Make plots."""
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

def cifar10_loader(root, batch_size):
    """
    CIFAR10 dataset loader.

    Args:
        root: data root directory
        batch_size: batch size to load testset
    """
    transform_test = transforms.Compose([
        transforms.CenterCrop(32),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    testset = torchvision.datasets.CIFAR10(
        root=root, train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=8)

    return testloader




#Main

testloader = cifar10_loader(dataroot, batch_size_test)

if section == "option1":
    # Perturb Real Images
    real_images_perturb(testloader, modelroot, batch_size_test, cuda)
    print("Perturb Real Images done ...")

elif section == "option2":
    #model = torch.load('/u/training/tra272/scratch/hw7/part_1/new/tempD.model')
    model = torch.load('/u/training/tra272/scratch/hw7/hw7_2nd/model/tempD.model')
    fig_name = '_with_G'
    # Synthetic Images Maximizing Classification Output
    max_class_function(testloader, model, cuda, fig_name)
    checkpoint = torch.load('/u/training/tra272/scratch/hw7/hw7_2nd/model/cifar10.pth')
    model = checkpoint['model']
    fig_name = '_without_G'
    # Synthetic Images Maximizing Classification Output
    max_class_function(testloader, model, cuda, fig_name)
    print("Synthetic Images Maximizing Classification Output done ...")

elif section == "option3":

    model = discriminatorBottom(extract_features=2)
    checkpoint = torch.load("/u/training/tra272/scratch/hw7/hw7_2nd/model/cifar10.pth")
    pre_trained_model = checkpoint['model']
    model.load_state_dict(pre_trained_model.state_dict())

    extract_features = 2
    fig_name = str(extract_features) +'_without_G'
    # Synthetic Features Maximizing Features at Various Layers
    max_features_function(testloader, model, batch_size_test, cuda, extract_features, fig_name)

    # with-G
    model = discriminatorBottom(extract_features=2)
    pre_trained_model = torch.load("/u/training/tra272/scratch/hw7/hw7_2nd/model/tempD.model")
    model.load_state_dict(pre_trained_model.state_dict())

    extract_features = 2
    fig_name = str(extract_features) + '_with_G'
    # Synthetic Features Maximizing Features at Various Layers
    max_features_function(testloader, model, batch_size_test, cuda, extract_features, fig_name)

    ####################

    model = discriminatorBottom(extract_features=4)
    checkpoint = torch.load("/u/training/tra272/scratch/hw7/hw7_2nd/model/cifar10.pth")
    pre_trained_model = checkpoint['model']
    model.load_state_dict(pre_trained_model.state_dict())

    extract_features = 4
    fig_name = str(extract_features) +'_without_G'
    # Synthetic Features Maximizing Features at Various Layers
    max_features_function(testloader, model, batch_size_test, cuda, extract_features, fig_name)

    # with-G
    model = discriminatorBottom(extract_features=4)
    pre_trained_model = torch.load("/u/training/tra272/scratch/hw7/hw7_2nd/model/tempD.model")
    model.load_state_dict(pre_trained_model.state_dict())

    extract_features = 4
    fig_name = str(extract_features) + '_with_G'
    # Synthetic Features Maximizing Features at Various Layers
    max_features_function(testloader, model, batch_size_test, cuda, extract_features, fig_name)

    print("Synthetic Features Maximizing Features at Various Layers done ...")
