import torch
import numpy as np
import cv2
import time
import math
import torch.nn as nn
def edge_extraction(gen_frames,use_cuda):
    
    # calculate the loss for each scale
    # create filters [-1, 1] and [[1],[-1]] for diffing to the left and down respectively.
    channels = gen_frames.shape[1]
    pos = torch.from_numpy(np.identity(channels))     # 3 x 3
    neg = -1 * pos
    filter_x = torch.cat([neg, pos], 0).unsqueeze(0)
    filter_y = torch.cat([neg, pos], 1).unsqueeze(0)
    # strides = [1, 1, 1, 1]  # stride of (1, 1)
    # padding = 'SAME'
    conv_x = nn.Conv2d(1,1,kernel_size=(2,1),stride=1,bias=False,padding=0)
    conv_y = nn.Conv2d(1,1,kernel_size=(1,2),stride=1,bias=False,padding=0)
    conv_x.requires_grad = False
    conv_y.requires_grad = False
    x_pad=nn.ZeroPad2d((0,0,1,0))
    y_pad=nn.ZeroPad2d((1,0,0,0))
    if use_cuda:
        filter_x=filter_x.cuda()
        filter_y=filter_y.cuda()
        conv_x=conv_x.cuda()
        conv_y=conv_y.cuda()
        x_pad=x_pad.cuda()
        y_pad=y_pad.cuda()
    conv_x.weight.data = filter_x.unsqueeze(0).float()
    conv_y.weight.data = filter_y.unsqueeze(0).float()

    # a=conv_x(gen_frames)
    gen_dx = torch.abs(conv_x(x_pad(gen_frames)))
    gen_dy = torch.abs(conv_y(y_pad(gen_frames)))

    edge = gen_dx ** 1 + gen_dy ** 1
    edge_clip  = torch.clamp(edge, 0, 1)
    # condense into one tensor and avg
    return edge_clip
def seammask_extraction(mask,use_cuda):

    seam_mask = edge_extraction(torch.unsqueeze(torch.mean(mask, axis=1),1),use_cuda)
    filters = torch.from_numpy(np.array([[1.0,1.0,1.0],[1.0,1.0,1.0],[1.0,1.0,1.0]])).unsqueeze(0).unsqueeze(0).float()
    conv_1 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_2 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_3 = nn.Conv2d(1,1,kernel_size=3,stride=1,bias=False,padding=1)
    conv_2.requires_grad = False
    conv_2.requires_grad = False
    conv_3.requires_grad = False

    if use_cuda:
        conv_1=conv_1.cuda()
        conv_2=conv_2.cuda()
        conv_3=conv_3.cuda()
        filters=filters.cuda()
    conv_1.weight.data = filters
    conv_2.weight.data = filters
    conv_3.weight.data = filters
    test_conv1 =conv_1(seam_mask)
    test_conv1 = torch.clamp(test_conv1, 0, 1)
    test_conv2 =conv_2(test_conv1)
    test_conv2 = torch.clamp(test_conv2, 0, 1)
    test_conv3 =conv_3(test_conv2)
    test_conv3 = torch.clamp(test_conv3, 0, 1)
    return test_conv3

def rgb2ycrcb(rgb):
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    # cr = (r - y) * 0.713 + 128
    # cb = (b - y) * 0.564 + 128
    cr = ((r - y)*255 * 0.713 + 128)/255
    cb = ((b - y)*255 * 0.564 + 128)/255
    ycrcb = torch.zeros_like(rgb)
    ycrcb[:, 0], ycrcb[:, 1], ycrcb[:, 2] = y, cr, cb
    return ycrcb

def DLT_solve(src_p, off_set):
    bs, _ = src_p.shape
    divide = int(np.sqrt(len(src_p[0])/2)-1)# divide=1 
    row_num = (divide+1)*2# row_num = 4，可能是看几边形吧
    for i in range(divide):
        for j in range(divide):
            h4p = src_p[:,[ 2*j + row_num*i, 2*j + row_num*i + 1, 
                    2*(j+1) + row_num*i, 2*(j+1) + row_num*i + 1, 
                    2*(j+1) + row_num*i + row_num, 2*(j+1) + row_num*i + row_num + 1,
                    2*j + row_num*i + row_num, 2*j + row_num*i + row_num+1]].reshape(bs, 1, 4, 2)  
            pred_h4p = off_set[:,[2*j+row_num*i, 2*j+row_num*i+1, 
                    2*(j+1)+row_num*i, 2*(j+1)+row_num*i+1, 
                    2*(j+1)+row_num*i+row_num, 2*(j+1)+row_num*i+row_num+1,
                    2*j+row_num*i+row_num, 2*j+row_num*i+row_num+1]].reshape(bs, 1, 4, 2)
            if i+j==0:
                src_ps = h4p
                off_sets = pred_h4p
            else:
                src_ps = torch.cat((src_ps, h4p), axis = 1)    
                off_sets = torch.cat((off_sets, pred_h4p), axis = 1)

    bs, n, h, w = src_ps.shape

    N = bs*n #1*1=1

    src_ps = src_ps.reshape(N, h, w)#(1,4,2)
    off_sets = off_sets.reshape(N, h, w)#(1,4,2)

    dst_p = src_ps + off_sets# 直接加偏移量,新的图像四边形
    # print(dst_p)
    ones = torch.ones(N, 4, 1) #(1,4,1)
    if torch.cuda.is_available():
        ones = ones.cuda()
    xy1 = torch.cat((src_ps, ones), 2)#(1,4,3) 
    zeros = torch.zeros_like(xy1)#(1,4,3)
    if torch.cuda.is_available():
        zeros = zeros.cuda()

    xyu, xyd = torch.cat((xy1, zeros), 2), torch.cat((zeros, xy1), 2)#(1,4,6)
    M1 = torch.cat((xyu, xyd), 2).reshape(N, -1, 6)
    M2 = torch.matmul(
        dst_p.reshape(-1, 2, 1), 
        src_ps.reshape(-1, 1, 2),
    ).reshape(N, -1, 2)

    A = torch.cat((M1, -M2), 2)
    b = dst_p.reshape(N, -1, 1)

    Ainv = torch.inverse(A)
    h8 = torch.matmul(Ainv, b).reshape(N, 8)
 
    H = torch.cat((h8, ones[:,0,:]), 1).reshape(N, 3, 3)
    H = H.reshape(bs, n, 3, 3)
    return H