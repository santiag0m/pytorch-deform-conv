from __future__ import absolute_import, division

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_deform_conv.layers import ConvOffset2D

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()

        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv11(x))
        x = self.bn11(x)

        x = F.relu(self.conv12(x))
        x = self.bn12(x)

        x = F.relu(self.conv21(x))
        x = self.bn21(x)

        x = F.relu(self.conv22(x))
        x = self.bn22(x)

        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))#
        x = F.softmax(x)
        return x

class DeformConvNet(nn.Module):
    def __init__(self, getForwardI = True):
        super(DeformConvNet, self).__init__()
        self.getForwardInfo = getForwardI
        # conv11
        self.conv11 = nn.Conv2d(1, 32, 3, padding=1)
        self.bn11 = nn.BatchNorm2d(32)

        # conv12
        self.offset12 = ConvOffset2D(32)
        self.conv12 = nn.Conv2d(32, 64, 3, padding=1, stride=2)
        self.bn12 = nn.BatchNorm2d(64)

        # conv21
        self.offset21 = ConvOffset2D(64)
        self.conv21 = nn.Conv2d(64, 128, 3, padding= 1)
        self.bn21 = nn.BatchNorm2d(128)

        # conv22
        self.offset22 = ConvOffset2D(128)
        self.conv22 = nn.Conv2d(128, 128, 3, padding=1, stride=2)
        self.bn22 = nn.BatchNorm2d(128)

        # out
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        
        forward_info = []
        
        shape = x.shape
        x = F.relu(self.conv11(x))
        x = self.bn11(x)
        if self.getForwardInfo:
            conv1_layer = ('conv', self.conv11, shape)
            forward_info.append(conv1_layer)
        
        shape = x.shape
        x, offsets = self.offset12(x)
        if self.getForwardInfo:
            offset1_layer = ('offset', offsets)
            forward_info.append(offset1_layer)
        
        shape = x.shape
        x = F.relu(self.conv12(x))
        x = self.bn12(x)
        if self.getForwardInfo:
            conv2_layer = ('conv', self.conv12, shape)
            forward_info.append(conv2_layer)
        
        shape = x.shape
        x, offsets = self.offset21(x)
        if self.getForwardInfo:
            offset2_layer = ('offset', offsets)
            forward_info.append(offset2_layer)
        
        shape = x.shape
        x = F.relu(self.conv21(x))
        x = self.bn21(x)
        if self.getForwardInfo:
            conv3_layer = ('conv', self.conv21, shape)
            forward_info.append(conv3_layer)
        
        shape = x.shape
        x, offsets = self.offset22(x)
        if self.getForwardInfo:
            offset3_layer = ('offset', offsets)
            forward_info.append(offset3_layer)
        
        x = F.avg_pool2d(x, kernel_size=[x.size(2), x.size(3)])
        x = self.fc(x.view(x.size()[:2]))
        x = F.softmax(x)
        
        return x, forward_info

    def freeze(self, module_classes):
        '''
        freeze modules for finetuning
        '''
        for k, m in self._modules.items():
            if any([type(m) == mc for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = False

    def unfreeze(self, module_classes):
        '''
        unfreeze modules
        '''
        for k, m in self._modules.items():
            if any([isinstance(m, mc) for mc in module_classes]):
                for param in m.parameters():
                    param.requires_grad = True

    def parameters(self):
        return filter(lambda p: p.requires_grad, super(DeformConvNet, self).parameters())

def get_cnn():
    return ConvNet()

def get_deform_cnn(trainable=True, freeze_filter=[nn.Conv2d, nn.Linear]):
    model = DeformConvNet()
    if not trainable:
        model.freeze(freeze_filter)
    return model
