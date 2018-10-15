from __future__ import absolute_import, division
# %env CUDA_VISIBLE_DEVICES=0

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

import torch.nn as nn
import torchvision
import tqdm

from torch_deform_conv.layers import ConvOffset2D
from torch_deform_conv.cnn import get_cnn, get_deform_cnn
from torch_deform_conv.utils import transfer_weights


batch_size_train = 32
batch_size_test = 100
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),\
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
dataset = torchvision.datasets.MNIST('MNIST/', train=True, download=True,transform=transform)
train_loader = torch.utils.data.DataLoader(dataset,batch_size=batch_size_train, shuffle=True)

test_data = torchvision.datasets.MNIST('MNIST/', train=False, download=True,transform=transform)
test_loader = torch.utils.data.DataLoader(test_data ,batch_size=batch_size_test, shuffle=True)

transform=torchvision.transforms.Compose([torchvision.transforms.RandomRotation(180),\
                                          torchvision.transforms.ToTensor(),\
                                          torchvision.transforms.Normalize((0.1307,), (0.3081,))])
test_data_rot = torchvision.datasets.MNIST('MNIST/', train=False, download=True,transform=transform)
test_loader_rot = torch.utils.data.DataLoader(test_data_rot ,batch_size=batch_size_test, shuffle=True)

def train(model, data_loader, epoch):
    model.train()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TRAIN] Epoch: {}".format(epoch)):
        data = data.float().cuda(); data = Variable(data)
        target = target.cuda(); target = Variable(target)

        output, forwa = model(data)
        optimizer.zero_grad()
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    print("Loss: %0.3f | Acc: %0.2f"%(np.array(loss_cum).mean(), float(Acc*100)/len(data_loader.dataset)))

def test(model, data_loader, epoch):
    model.eval()
    loss_cum = []
    Acc = 0
    for batch_idx, (data,target) in tqdm.tqdm(enumerate(data_loader), total=len(data_loader), desc="[TEST] Epoch: {}".format(epoch)):
        data = data.cuda(); data = Variable(data, volatile=True)
        target = target.cuda(); target = Variable(target, volatile=True)
        output, forwa = model(data)
        loss = F.cross_entropy(output, target)   
        loss_cum.append(loss.data.cpu()[0])
        _, arg_max_out = torch.max(output.data.cpu(), 1)
        Acc += arg_max_out.long().eq(target.data.cpu().long()).sum()
    Acc = float(Acc*100)/len(data_loader.dataset)
    lossi = np.array(loss_cum).mean()
    print("Loss Test: %0.3f | Acc Test: %0.2f"%(np.array(loss_cum).mean(), Acc))
    return Acc, lossi
# ---
# Normal CNN


#model = get_cnn()
#model = model.cuda()
#optimizer = optim.Adam(model.parameters(), lr=1e-3)
#for epoch in range(10):
#    test(model, test_loader, epoch)
#    train(model, train_loader, epoch)
#
#
#torch.save(model, 'models/cnn.th')
#
## ---
## Evaluate normal CNN
#
#print('Evaluate normal CNN')
#model_cnn = torch.load('models/cnn.th')
#
#test(model_cnn, test_loader, epoch)
## 99.27%
#test(model_cnn, test_loader_rot, epoch)
## 58.83%
#
## ---
# Deformable CNN

print('Finetune deformable CNN (ConvOffset2D and BatchNorm)')
model = get_deform_cnn(trainable=False)
model = model.cuda()
#transfer_weights(model_cnn, model)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
for epoch in range(10):
    test(model, test_loader, epoch)
    train(model, train_loader, epoch)

torch.save(model, 'models/deform_cnn.th')

# ---
# Evaluate deformable CNN

print('Evaluate deformable CNN')
model = torch.load('models/deform_cnn.th')

test(model, test_loader, epoch)
test(model, test_loader_rot, epoch)