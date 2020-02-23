#Sayantan Dutta
# Discriminator, Generator Classes

import torch
import torch.optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torch.nn.functional as F
import torchvision
import torch.utils.data
import torchvision.transforms as transforms
num_classes = 10


# Define the discriminator
class discriminator(nn.Module):
    def __init__(self):
        super(discriminator, self).__init__()
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
        x = F.leaky_relu(self.layer_norm3(self.conv3(x)))
        x = F.leaky_relu(self.layer_norm4(self.conv4(x)))
        x = F.leaky_relu(self.layer_norm5(self.conv5(x)))
        x = F.leaky_relu(self.layer_norm6(self.conv6(x)))
        x = F.leaky_relu(self.layer_norm7(self.conv7(x)))
        x = F.leaky_relu(self.layer_norm8(self.conv8(x)))

        x = F.max_pool2d(x, kernel_size=4, padding=0, stride=4)

        x = x.view(x.size(0), -1)
        out1 = self.fc1(x)
        out10 = self.fc10(x)

        return (out1, out10)


# Define the generator
class generator(nn.Module):
    def __init__(self):
        super(generator, self).__init__()
        self.fc1 = nn.Linear(100, 196 * 4 * 4)
        self.conv1 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv6 = nn.ConvTranspose2d(196, 196, kernel_size=3, stride=1, padding=1)
        self.conv7 = nn.ConvTranspose2d(196, 196, kernel_size=4, stride=2, padding=1)
        self.conv8 = nn.ConvTranspose2d(196, 3, kernel_size=3, stride=1, padding=1)
        self.batchnorm0 = nn.BatchNorm1d(196 * 4 * 4)
        self.batchnorm1 = nn.BatchNorm2d(196)
        self.batchnorm2 = nn.BatchNorm2d(196)
        self.batchnorm3 = nn.BatchNorm2d(196)
        self.batchnorm4 = nn.BatchNorm2d(196)
        self.batchnorm5 = nn.BatchNorm2d(196)
        self.batchnorm6 = nn.BatchNorm2d(196)
        self.batchnorm7 = nn.BatchNorm2d(196)

    def forward(self, x):

        x = self.batchnorm0(self.fc1(x))
        x = x.reshape(x.size()[0], 196, 4, 4)
        x = F.relu(self.batchnorm1(self.conv1(x)))
        x = F.relu(self.batchnorm2(self.conv2(x)))
        x = F.relu(self.batchnorm3(self.conv3(x)))
        x = F.relu(self.batchnorm4(self.conv4(x)))
        x = F.relu(self.batchnorm5(self.conv5(x)))
        x = F.relu(self.batchnorm6(self.conv6(x)))
        x = F.relu(self.batchnorm7(self.conv7(x)))
        x = self.conv8(x)

        x = nn.Tanh()(x)

        return x
