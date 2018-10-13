# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 12:21:03 2018

@author: Santiago
"""

import torch

def coord_grid(h, w):
    x_coord = [list(range(w))]*h
    x_coord = torch.tensor(x_coord)
    y_coord = [list(range(h))]*w
    y_coord = torch.tensor(y_coord).t()
    return x_coord, y_coord

def get_input_size(h, w, layer_info):
    k = layer_info[1]; k = [k]*2 if isinstance(k, int) else k
    s = layer_info[2]; s = [s]*2 if isinstance(s, int) else s
    p = layer_info[3]; p = [p]*2 if isinstance(p, int) else p
    d = layer_info[4]; d = [d]*2 if isinstance(d, int) else d
    
    h_in = ((h-1)*s[0])-(2*p[0]-d[0]*(k[0]-1)-1)
    h_in = int(h_in+1)
    w_in = ((w-1)*s[1])-(2*p[1]-d[1]*(k[1]-1)-1)
    w_in = int(w_in+1)
    
    return h_in, w_in
    
def get_conv_points(x, y, layer_info):
    k = layer_info[1]; k = [k]*2 if isinstance(k, int) else k
    s = layer_info[2]; s = [s]*2 if isinstance(s, int) else s
    p = layer_info[3]; p = [p]*2 if isinstance(p, int) else p
    d = layer_info[4]; d = [d]*2 if isinstance(d, int) else d
    h = layer_info[5][-2]
    w = layer_info[5][-1]
    
    y_pos = [y*s[0]-p[0]] # Top
    x_pos = [x*s[1]-p[0]] # Left
    
    for i in range(1,k[0]): # Rows
        temp_y = y_pos[0] + i*d[0]
        y_pos.append(temp_y)
        
    for j in range(1, k[1]): # Columns
        temp_x = x_pos[0] + j*d[1]
        x_pos.append(temp_x)
    
    pairs = list(zip(x_pos, y_pos))
    pairs = [pos for pos in pairs if pos[0]>=0 and pos[1]>=0] # Remove padded
    pairs = [pos for pos in pairs if pos[0]<w and pos[1]<h] # Remove padded
    x_points = [pos[0] for pos in pairs]
    y_points = [pos[1] for pos in pairs]
    
    return x_points, y_points

def get_points(forward_pass):
    """
    Returns a binary matrix for relevant pixels in
    the image based on a list of forward_pass info 
    for different layer types.
    
    Valid layer types are:
        'offset' - (type, offset_map)
        'conv' - (type, kernel_size, stride, padding, dilation, input_shape)
        'pool' - (type, input_shape, indices)
    """
    deform_modules = []
    begin = 0
    for i, t in enumerate(forward_pass):
        if t[0] == 'offset':
            deform_modules.append(forward_pass[begin:i+1])
            begin = i
    
    if len(deform_modules)<=0:
        raise Exception('There must be at least 1 deformable layer')
    
    x_points = None
    y_points = None
    
    for module in reversed(deform_modules):
        for layer in reversed(module):
            if t[0] == 'offset':
                offset = t[1]
                c = offset.shape[0]
                h = offset.shape[1]
                w = offset.shape[2]
                # Create default positioning grid
                x_coord, y_coord = coord_grid(h, w)
                x_coord = torch.cat([x_coord.unsqueeze(0)]*c, 0)
                y_coord = torch.cat([y_coord.unsqueeze(0)]*c, 0)
                # Add offset
                x_coord = x_coord + offset[:,:,:,0]
                y_coord = y_coord + offset[:,:,:,1]
                
                if x_points is not None:
                    # Index relevant points if provided
                    for i in c:
                        temp_x = [x_coord[i, x, y] for x, y in zip(x_points, y_points)]
                        temp_y = [y_coord[i, x, y] for x, y in zip(x_points, y_points)]
                        x_points = temp_x
                        y_points = temp_y
                else:
                    x_points = x_coord.view(-1).tolist()
                    y_points = y_coord.view(-1).tolist()
                
            elif t[0] == 'conv':
                h = t[5][-2]
                w = t[5][-1]
                temp_x = []
                temp_y = []
                
                if x_points is None:
                    x_coord, y_coord = coord_grid(h, w)
                    x_points = x_coord.view(-1).tolist()
                    y_points = y_coord.view(-1).tolist()
                
                for i in range(len(x_points)):
                    # Get positions for relevant points
                    x, y = get_conv_points(x_points[i], y_points[i], t)
                    temp_x += x
                    temp_y += y
                x_points = temp_x
                y_points = temp_y
            elif t[0] == 'pool':
                c = t[1][1]
                h = t[1][2]
                w = t[1][3]
                idxs = t[2]
                temp_x = []
                temp_y = []
                
                if x_points is None:
                    x_coord, y_coord = coord_grid(idxs.shape[2], idxs.shape[3])
                    x_points = x_coord.view(-1).tolist()
                    y_points = y_coord.view(-1).tolist()
                
                for i in c: # No batch support
                    temp_x = [idxs[0, i, x, y]%w for x, y in zip(x_points, y_points)]
                    temp_y = [idxs[0, i, x, y]//w for x, y in zip(x_points, y_points)]
                    x_points = temp_x
                    y_points = temp_y
                
            else:
                raise Exception('"'+t[0]+'" is not valid layer type')
    return x_points, y_points
                    
                
                