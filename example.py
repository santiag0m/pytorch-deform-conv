# -*- coding: utf-8 -*-

import torch.nn as nn
from torch_deform_conv.layers import ConvOffset2D

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1,1,3)
        self.pool = nn.MaxPool2d(2,2,return_indices=True)
        self.defconv = ConvOffset2D(1, return_offsets=True)
        self.conv2 = nn.Conv2d(1,1,3)
        
    def forward(self, x, getForwardInfo = False):
        if getForwardInfo:
            # Initialize
            forward_info = []
            
        # First Conv layer
        shape = x.shape
        x = self.conv1(x)
        conv1_layer = ('conv', self.conv1, shape)
        forward_info.append(conv1_layer)
        
        # Pooling layer
        shape = x.shape
        x, idxs = self.pool(x)
        pool_layer = ('pool', idxs, shape)
        forward_info.append(pool_layer)
        
        # Deformable conv layer
        shape = x.shape
        x, offsets = self.defconv(x)
        offset_layer = ('offset', offsets)
        forward_info.append(offset_layer)
        
        # Second Conv layer
        shape = x.shape
        x = self.conv2(x)
        conv2_layer = ('conv', self.conv2, shape)
        forward_info.append(conv2_layer)
        
        return x, forward_info
        
if __name__ == '__main__':
    import torch
    from visualization import get_points
    x = torch.randn(1,1,52,52)
    model = Net()
    y, forward_info = model(x,False)
    x_points, y_points = get_points(forward_info)
