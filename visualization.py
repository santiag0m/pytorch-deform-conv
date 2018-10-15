# -*- coding: utf-8 -*-

import torch
#import matplotlib.pyplot as plt

def coord_grid(h, w):
    x_coord = [list(range(w))]*h
    x_coord = torch.tensor(x_coord, dtype=torch.float).contiguous()
    y_coord = [list(range(h))]*w
    y_coord = torch.tensor(y_coord, dtype=torch.float).t().contiguous()
    return x_coord, y_coord

    
def get_conv_points(x, y, layer_info):

    k = layer_info[1].kernel_size; k = [k]*2 if isinstance(k, int) else k
    s = layer_info[1].stride; s = [s]*2 if isinstance(s, int) else s
    p = layer_info[1].padding; p = [p]*2 if isinstance(p, int) else p
    d = layer_info[1].dilation; d = [d]*2 if isinstance(d, int) else d
    h = layer_info[2][-2]
    w = layer_info[2][-1]
    
    
    y_pos = []
    x_pos = []
    
    for i in range(k[0]): # Rows
        if i == 0:
            temp_y = y*s[0]-p[0] # Top
        else:
            temp_y = y_pos[0] + i*d[0]
        for j in range(k[1]): # Columns
            if j == 0:
                temp_x = x*s[1]-p[1] # Left
            else:
                temp_x = x_pos[0] + j*d[1]
            x_pos.append(temp_x)
            y_pos.append(temp_y)
        
    
    
    pairs = list(zip(x_pos, y_pos))
    pairs = [pos for pos in pairs if pos[0]>=0 and pos[1]>=0] # Remove padded
    pairs = [pos for pos in pairs if pos[0]<w and pos[1]<h] # Remove padded
    pairs = set(pairs)
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
        'conv' - (type, layer, input_shape)
        'pool' - (type, indices,  input_shape)
    """
    
    x_points = None
    y_points = None
    
    for t in reversed(forward_pass):
        if t[0] == 'offset':
            print('offset')
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
                x_buffer = []
                y_buffer = []
                for i in range(c):
                    temp_x = [x_coord[i, int(x), int(y)].data for x, y in zip(x_points, y_points)]
                    temp_y = [y_coord[i, int(x), int(y)].data for x, y in zip(x_points, y_points)]
                    temp_x = [x.unsqueeze(0) for x in temp_x]
                    temp_y = [y.unsqueeze(0) for y in temp_y]
                    temp_x = torch.cat(temp_x,0).tolist()
                    temp_y = torch.cat(temp_y,0).tolist()
                    x_buffer += temp_x
                    y_buffer += temp_y
                x_points = x_buffer
                y_points = y_buffer
            else:
                x_points = x_coord.view(-1).tolist()
                y_points = y_coord.view(-1).tolist()
                
            
#            plt.close('all')
#            plt.scatter(x_points, y_points)
#            plt.title('Offset')
#            plt.pause(1)
            
        elif t[0] == 'conv':
            print('conv')
            h = t[2][-2]
            w = t[2][-1]
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
            
#            plt.close('all')
#            plt.scatter(x_points, y_points)
#            plt.title('Conv')
#            plt.pause(1)
            
        elif t[0] == 'pool':
            print('pool')
            h = t[2][2]
            w = t[2][3]
            idxs = t[1]
            temp_x = []
            temp_y = []
            
            if x_points is None:
                x_coord, y_coord = coord_grid(idxs.shape[2], idxs.shape[3])
                x_points = x_coord.view(-1).tolist()
                y_points = y_coord.view(-1).tolist()
            else:
                pairs = [(x, y) for x, y in zip(x_points, y_points) if (x<idxs.shape[3] and y<idxs.shape[2])]
                pairs = set(pairs)
                pairs = [p for p in pairs if p[0]>0 and p[1]>0]
                x_points = [p[0] for p in pairs]
                y_points = [p[1] for p in pairs]
            
            
            x_buffer = []
            y_buffer = []
            for i in range(idxs.shape[1]): # No batch support
                temp_x = [(idxs[0, i, int(x), int(y)].data)%w for x, y in zip(x_points, y_points)]
                temp_y = [(idxs[0, i, int(x), int(y)].data)//w for x, y in zip(x_points, y_points)]
                temp_x = [x.unsqueeze(0) for x in temp_x]
                temp_y = [y.unsqueeze(0) for y in temp_y]
                temp_x = torch.cat(temp_x,0).tolist()
                temp_y = torch.cat(temp_y,0).tolist()
                x_buffer += temp_x
                y_buffer += temp_y
            x_points = x_buffer
            y_points = y_buffer
                
#            plt.close('all')
#            plt.scatter(x_points, y_points)
#            plt.title('Pooling')
#            plt.pause(1)
            
        else:
            raise Exception('"'+t[0]+'" is not valid layer type')
    return x_points, y_points
