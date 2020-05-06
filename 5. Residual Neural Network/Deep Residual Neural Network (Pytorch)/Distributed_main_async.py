import torch
import torch.utils.data
import torch.nn as nn
import h5py
import time
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pandas as pd
import torch.distributed as dist



import os
import subprocess
from mpi4py import MPI





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













# 3x3 convolution
start_time = time.time()
# Hyper-parameters
num_epochs = 60
learning_rate = 0.001
batch_size = 128
dropout_prob = 0.2
dropout_prob_1d = 0.5

torch.manual_seed(0)

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


# trainset = torchvision.datasets.CIFAR100(root='/home/thanque/Documents/IE534/HW4', train=True, download=False, transform=transform_train)
trainset = torchvision.datasets.CIFAR100(root='~/scratch/', train=True, download=False, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR100(root='/home/thanque/Documents/IE534/HW4', train=False, download=False, transform=transform_test)
testset = torchvision.datasets.CIFAR100(root='~/scratch/', train=False, download=False, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

print('Finish dataloader and time is', time.time()-start_time)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=100):
        self.inplanes = 32
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1,
                              bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 32, layers[0],  stride=1)
        self.layer2 = self._make_layer(block, 64, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 128, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=2)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes)
        self.dropout = torch.nn.Dropout2d(p=dropout_prob)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #x = self.maxpool(x)
        #print(x.shape)
        x = self.dropout(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        #x = self.avgpool(x)
        x = F.max_pool2d(x, 4)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
model =  ResNet(BasicBlock,[2,4,4,2])


for param in model.parameters():
    tensor0 = param.data
    dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
    param.data = tensor0/np.sqrt(np.float(num_nodes))




model.cuda()

tensor_buffer = {} 
tag_dict = {}
counter = 0
for name,param in model.named_parameters():
    tensor_buffer[name] = torch.zeros(param.data.shape).cpu()
    tag_dict[name] = counter
    counter += 1



# for name in tensor_buffer.keys():
#     print(name,tensor_buffer[name])

# exit()




criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.RMSprop(model.parameters(), lr = learning_rate)

scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10,gamma=0.5)


train_acc_list = []
test_acc_list = []
running_time_list = []

start_time = time.time()
for epoch in range(num_epochs):
    train_accu = []
    model.train()
    for batch_idx, (X_train_batch, Y_train_batch) in enumerate(trainloader):
        X_train_batch,Y_train_batch = Variable(X_train_batch).cuda(),Variable(Y_train_batch).cuda()
        # X_train_batch,Y_train_batch = Variable(X_train_batch),Variable(Y_train_batch)
        output = model(X_train_batch)
        loss = criterion(output, Y_train_batch)
        optimizer.zero_grad()
        # start_time = time.time()
        loss.backward()
        # print(time.time()-start_time)
        # exit()
        
        # #https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/module.py
        #https://github.com/pytorch/pytorch/blob/master/torch/distributed/distributed_c10d.py

        # loss.backward()
        # for name,param in model.named_parameters():
        #     param.grad.data = param.grad.data + torch.tensor(88.0).cuda()
        #     print(name,param.grad)
        # exit()
   
            # break
            # param.grad = param.grad + torch.tensor(100).cuda()
            # print(param.grad.data)

        #         tensor0 = param.grad.data.cpu()


        req = None
        if rank != 0:
            # print(rank,'step 1')
            for name,param in model.named_parameters():
                tensor0 = param.grad.data.cpu()
                # req = dist.isend(tensor=tensor0, dst=0,tag=tag_dict[name])
                req = dist.isend(tensor=tensor0, dst=0)

                req.wait()
            # print(rank,'step 2')
                
            # print(rank,'step 3')
            for name,param in model.named_parameters():
                # req = dist.irecv(tensor=tensor_buffer[name], src=0,tag=tag_dict[name])
                req = dist.irecv(tensor=tensor_buffer[name], src=0)
                param.data = tensor_buffer[name].cuda()
                # param.data = tensor_buffer[name]
            # print(rank,'step 4')
                req.wait()
            # print(rank,'step 5')
        else:
            for ii in range(1,num_nodes):
                # print(rank,'step 1')
                for name,param in model.named_parameters():
                    # tensor0 = param.grad.data.cpu()
                    # req = dist.isend(tensor=tensor0, dst=0)
                    
                    # req = dist.irecv(tensor=tensor_buffer[name],src=ii,tag=tag_dict[name])
                    req = dist.irecv(tensor=tensor_buffer[name],src=ii)
                    # print(name,req)
                    param.grad.data = tensor_buffer[name].cuda()
                    # param.grad.data = tensor_buffer[name]
                # print(rank,'step 2')
                    req.wait()
                optimizer.step()
                # print(rank,'step 3')
                for name,param in model.named_parameters():
                    tensor0 = param.data.cpu()
                    # req = dist.isend(tensor=tensor0, dst=ii,tag=tag_dict[name])
                    req = dist.isend(tensor=tensor0, dst=ii)
                    req.wait()
                # print(rank,'step 4')
                
                # print(rank,'step 5')
    print('rank',rank,'epoch',epoch,'time',time.time()-start_time)

    if rank == 0:
        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(Y_train_batch.data).sum() ) /float(batch_size))*100.0
        train_acc = accuracy

        model.eval()
        test_accu = []
        for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
            X_test_batch, Y_test_batch= Variable(X_test_batch,volatile=True).cuda(),Variable(Y_test_batch,volatile=True).cuda()
            output = model(X_test_batch)
            prediction = output.data.max(1)[1] 
            accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
            test_accu.append(accuracy)
        test_acc = np.mean(test_accu)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        running_time_list.append(time.time()-start_time)
        print('Rank',rank,'Epoch',epoch,'Train accuracy',train_acc,'Test accuracy',test_acc,'Running time',time.time()-start_time)
        
if rank == 0:

    df = pd.DataFrame({'train_acc':train_acc_list,
                    'test_acc':test_acc_list,
                    'time':running_time_list
        })
    df.to_csv('result_async.csv',index=False)
        
        #     if batch_idx % 1000 == 0:
        #         print(epoch,batch_idx,time.time()-start_time)
        #     for param in model.parameters():
        #         #print(param.grad.data)
        #         tensor0 = param.grad.data.cpu()
        #         dist.all_reduce(tensor0, op=dist.reduce_op.SUM)
        #         tensor0 /= float(num_nodes)
        #         param.grad.data = tensor0.cuda()    
        #     if batch_idx % 1000 == 0:
        #         print(epoch,batch_idx,time.time()-start_time)

        #     optimizer.step()
        #     prediction = output.data.max(1)[1]
        #     accuracy = ( float( prediction.eq(Y_train_batch.data).sum() ) /float(batch_size))*100.0
        #     train_accu.append(accuracy)
        # scheduler.step()
        # train_acc = np.mean(train_accu)
    # Test the model

        
        # model.eval()
        # test_accu = []
        # for batch_idx, (X_test_batch, Y_test_batch) in enumerate(testloader):
        #     X_test_batch, Y_test_batch= Variable(X_test_batch,volatile=True).cuda(),Variable(Y_test_batch,volatile=True).cuda()
        #     output = model(X_test_batch)
        #     prediction = output.data.max(1)[1] 
        #     accuracy = ( float( prediction.eq(Y_test_batch.data).sum() ) /float(batch_size))*100.0
        #     test_accu.append(accuracy)
        # test_acc = np.mean(test_accu)
        # print('Rank',rank,'Epoch',epoch,'Train accuracy',train_acc,'Test accuracy',test_acc,'Running time',time.time()-start_time)


