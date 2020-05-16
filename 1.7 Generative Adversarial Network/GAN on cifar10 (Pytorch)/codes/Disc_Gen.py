#Sayantan Dutta
#Discriminator with Generator


import torch
import torchvision
import torch.nn as nn
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.autograd as autograd
from torch.autograd import Variable
import numpy as np
import time
from DG_classes import discriminator, generator

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os


torch.cuda.is_available()
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 200
start_epoch = 160
num_classes = 10
batch_size = 128
learning_rate = 0.0001
resume = False

transform_train = transforms.Compose([
    transforms.RandomResizedCrop(32, scale=(0.7, 1.0), ratio=(1.0, 1.0)),
    transforms.ColorJitter(
        brightness=0.1*torch.randn(1),
        contrast=0.1*torch.randn(1),
        saturation=0.1*torch.randn(1),
        hue=0.1*torch.randn(1)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

transform_test = transforms.Compose([
    transforms.CenterCrop(32),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

trainset = torchvision.datasets.CIFAR10(root='./', train=True, download=True, transform=transform_train)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=8)

testset = torchvision.datasets.CIFAR10(root='./', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=8)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

'''Training a GAN takes significantly longer than training just the discriminator. For each iteration, we will need to update the generator network and the discriminator network separately. The generator network requires a forward/backward pass through both the generator and the discriminator. The discriminator network requires a forward pass through the generator and two forward/backward passes for the discriminator (one for real images and one for fake images). On top of this, the model can be trained for a large number of iterations. I trained for 500 epochs.'''

def calc_gradient_penalty(netD, real_data, fake_data):
    DIM = 32
    LAMBDA = 10
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, int(real_data.nelement()/batch_size)).contiguous()
    alpha = alpha.view(batch_size, 3, DIM, DIM)
    alpha = alpha.cuda()

    fake_data = fake_data.view(batch_size, 3, DIM, DIM)
    interpolates = alpha * real_data.detach() + ((1 - alpha) * fake_data.detach())

    interpolates = interpolates.cuda()
    interpolates.requires_grad_(True)

    disc_interpolates, _ = netD(interpolates)

    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                              create_graph=True, retain_graph=True, only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
    return gradient_penalty

'''The above function is the gradient penalty described in the Wasserstein GAN section. Notice once again that only the fc1 output is used instead of the fc10 output. The returned gradient penalty will be used in the discriminator loss during optimization.'''

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

'''This function is used to plot a 10 by 10 grid of images scaled between 0 and 1. After every epoch, we will use a batch of noise saved at the start of training to see how the generator improves over time.'''

if resume:
    aD = torch.load('tempD.model')
    aG = torch.load('tempG.model')
else:
    aD = discriminator()
    aG = generator()



aD.cuda()
aG.cuda()

optimizer_g = torch.optim.Adam(aG.parameters(), lr=0.0001, betas=(0, 0.9))
optimizer_d = torch.optim.Adam(aD.parameters(), lr=0.0001, betas=(0, 0.9))

criterion = nn.CrossEntropyLoss()

'''Create the two networks and an optimizer for each. Note the non-default beta parameters. The first moment decay rate is set to 0. This seems to help stabilize training.'''

n_z = 100
n_classes = 10
np.random.seed(352)
label = np.asarray(list(range(10))*10)
noise = np.random.normal(0, 1, (100, n_z))
label_onehot = np.zeros((100, n_classes))
label_onehot[np.arange(100), label] = 1
noise[np.arange(100), :n_classes] = label_onehot[np.arange(100)]
noise = noise.astype(np.float32)

save_noise = torch.from_numpy(noise)
save_noise = Variable(save_noise).cuda()

'''This is a random batch of noise for the generator. n_z is set to 100 since this is the expected input for the generator. The noise is not entirely random. This creates an array label which is the repeated sequence 0-9 ten different times. This means the batch size is 100 and there are 10 examples for each class. The first 10 dimensions of the 100 dimension noise are set to be the one-hot representation of the label. This means a 0 is used in all spots except the index corresponding to a label where a 1 is located.'''

start_time = time.time()

# Train
for epoch in range(start_epoch, num_epochs):

    print("\n--- Epoch : %2d ---" % epoch)


    loss1 = []
    loss2 = []
    loss3 = []
    loss4 = []
    loss5 = []
    acc1 = []

    aG.train()
    aD.train()
	
	'''It is necessary to put the generator into train mode since it uses batch normalization. Technically the discriminator has no dropout or layer normalization meaning train mode should return the same values as test mode. However, it is good practice to keep it here in case we were to add dropout to the discriminator (which can improve results when training GANs).'''

    for group in optimizer_g.param_groups:
        for p in group['params']:
            state = optimizer_g.state[p]
            if('step' in state and state['step'] >= 1024):
                state['step'] = 1000

    for group in optimizer_d.param_groups:
        for p in group['params']:
            state = optimizer_d.state[p]
            if('step' in state and state['step'] >= 1024):
                state['step'] = 1000

    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(train_loader):

        if(Y_train_batch.shape[0] < batch_size):
            continue

        # train G
        gen_train = 1

        if((batch_idx % gen_train) == 0):
            for p in aD.parameters():
                p.requires_grad_(False)

            aG.zero_grad()

            label = np.random.randint(0, n_classes, batch_size)
            noise = np.random.normal(0, 1, (batch_size, n_z))
            label_onehot = np.zeros((batch_size, n_classes))
            label_onehot[np.arange(batch_size), label] = 1
            noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
            noise = noise.astype(np.float32)
            noise = torch.from_numpy(noise)
            noise = Variable(noise).cuda()
            fake_label = Variable(torch.from_numpy(label)).cuda()

            fake_data = aG(noise)
            gen_source, gen_class = aD(fake_data)

            gen_source = gen_source.mean()
            gen_class = criterion(gen_class, fake_label)

            gen_cost = -gen_source + gen_class
            gen_cost.backward()

            optimizer_g.step()
			
			'''The first portion of each iteration is for training the generator. Note the first if() statement. I set gen_train equal to 1 meaning the generator is trained every iteration just like the discriminator. Sometimes the discriminator is trained more frequently than the generator meaning gen_train can be set to something like 5. This seemed to matter more to stabilize training before the Wasserstein GAN loss function started getting used.

			The gradients for the discriminator parameters are turned off during the generator update as this saves GPU memory. Random noise is generated along with random labels to modify the noise. fake_data are the fake images coming from the generator. The discriminator provides two outputs: one from fc1 and one from fc10. The gen_source is the value from fc1 specifying if the discriminator thinks its input is real or fake. The discriminator wants this to be positive for real images and negative for fake images. Because of this, the generator wants to maximize this value (hence the negative since in the line calculating gen_cost). The gen_class output is from fc10 specifying which class the discriminator thinks the image is. Even though these are fake images, the noise contains a segment based on the label we want the generator to generate. Therefore, we combine these losses, perform a backward step, and update the parameters.'''


        for p in aD.parameters():
            p.requires_grad_(True)

        aD.zero_grad()

        # train D with G
        label = np.random.randint(0, n_classes, batch_size)
        noise = np.random.normal(0, 1, (batch_size, n_z))
        label_onehot = np.zeros((batch_size, n_classes))
        label_onehot[np.arange(batch_size), label] = 1
        noise[np.arange(batch_size), :n_classes] = label_onehot[np.arange(batch_size)]
        noise = noise.astype(np.float32)
        noise = torch.from_numpy(noise)
        noise = Variable(noise).cuda()
        fake_label = Variable(torch.from_numpy(label)).cuda()
        with torch.no_grad():
            fake_data = aG(noise)

        disc_fake_source, disc_fake_class = aD(fake_data)

        disc_fake_source = disc_fake_source.mean()
        disc_fake_class = criterion(disc_fake_class, fake_label)


        real_data = Variable(X_train_batch).cuda()
        real_label = Variable(Y_train_batch).cuda()

        disc_real_source, disc_real_class = aD(real_data)

        prediction = disc_real_class.data.max(1)[1]
        accuracy = (float(prediction.eq(real_label.data).sum()) / float(batch_size))*100.0

        disc_real_source = disc_real_source.mean()
        disc_real_class = criterion(disc_real_class, real_label)

        gradient_penalty = calc_gradient_penalty(aD, real_data, fake_data)

        disc_cost = disc_fake_source - disc_real_source + disc_real_class + disc_fake_class + gradient_penalty
        disc_cost.backward()
		
		'''The _source losses are for the Wasserstein GAN formulation. The discriminator is trying to maximize the scores for the real data and minimize the scores for the fake data. The `_class losses are for the auxiliary classifier wanting to correctly identify the class regardless of if the data is real or fake.
		During training, there are multiple values to print out and monitor.'''

        optimizer_d.step()
		'''The discriminator is being trained on two separate batches of data. The first is on fake data coming from the generator. The second is on the real data. Lastly, the gradient penalty function is called based on the real and fake data. This results in 5 separate terms for the loss function.'''

        # within the training loop
        loss1.append(gradient_penalty.item())
        loss2.append(disc_fake_source.item())
        loss3.append(disc_real_source.item())
        loss4.append(disc_real_class.item())
        loss5.append(disc_fake_class.item())
        acc1.append(accuracy)
        if((batch_idx % 50) == 0):
            print(epoch, batch_idx, 'Loss1:',"%.2f" % np.mean(loss1),
                                    'Loss2:,'"%.2f" % np.mean(loss2),
                                    'Loss3:',"%.2f" % np.mean(loss3),
                                    'Loss4:',"%.2f" % np.mean(loss4),
                                    'Loss5:',"%.2f" % np.mean(loss5),
                                    'Accuracy:',"%.2f" % np.mean(acc1))
									
		'''As mentioned previously, the discriminator is trying to minimize disc_fake_source and maximize disc_real_source. The generator is trying to maximize disc_fake_source. The output from fc1 is unbounded meaning it may not necessarily hover around 0 with negative values indicating a fake image and positive values indicating a positive image. It is possible for this value to always be negative or always be positive. The more important value is the difference between them on average. This could be used to determine a threshold for the discriminator considers to be real or fake.'''

    # Test
    aD.eval()
    with torch.no_grad():
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(test_loader):
            X_test_batch, Y_test_batch = Variable(X_test_batch).cuda(), Variable(Y_test_batch).cuda()

            with torch.no_grad():
                _, output = aD(X_test_batch)

            prediction = output.data.max(1)[1]  # first column has actual prob.
            accuracy = (float(prediction.eq(Y_test_batch.data).sum()) / float(batch_size))*100.0
            test_accu.append(accuracy)
            accuracy_test = np.mean(test_accu)
    print('Testing', accuracy_test, time.time()-start_time)


    with torch.no_grad():
        aG.eval()
        samples = aG(save_noise)
        samples = samples.data.cpu().numpy()
        samples += 1.0
        samples /= 2.0
        samples = samples.transpose(0, 2, 3, 1)
        aG.train()

    fig = plot(samples)
    plt.savefig('output/%s.png' % str(epoch).zfill(3), bbox_inches='tight')
    plt.close(fig)

    if(((epoch+1) % 1) == 0):
        torch.save(aG, 'tempG.model')
        torch.save(aD, 'tempD.model')
		
	'''After every epoch, the save_noise created before training can be used to generate samples to see how the generator improves over time. The samples from the generator are scaled between -1 and 1. The plot function expects them to be scaled between 0 and 1 and also expects the order of the channels to be (batch_size,w,h,3) as opposed to how PyTorch expects it.'''

# save
torch.save(aG, 'generator.model')
torch.save(aD, 'discriminator.model')
