'''---------------------------------------------------------------------------
IFCNN: A General Image Fusion Framework Based on Convolutional Neural Network
----------------------------------------------------------------------------'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
# from context_block import ContextBlock
from gcn import BasicUnit
def L1_norm(narry_a, narry_b):
    temp_abs_a = torch.abs(narry_a)
    temp_abs_b = torch.abs(narry_b)
    l1_a = torch.sum(temp_abs_a, dim=1)
    l1_b = torch.sum(temp_abs_b, dim=1)
    mask_value = l1_a + l1_b
    array_MASK_a = torch.unsqueeze(l1_a / mask_value,1)
    array_MASK_b =  torch.unsqueeze(l1_b / mask_value,1)
    resule_tf = array_MASK_a * narry_a + array_MASK_b * narry_b
    return resule_tf

# My Convolution Block
class ConvBlock(nn.Module):
    def __init__(self, inplane, outplane):
        super(ConvBlock, self).__init__()
        self.padding = (1, 1, 1, 1)
        self.conv = nn.Conv2d(inplane, outplane, kernel_size=3, padding=0, stride=1, bias=False)
        # self.bn = nn.BatchNorm2d(outplane)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.pad(x, self.padding, 'replicate')
        out = self.conv(out)
        # out = self.bn(out)
        out = self.relu(out)
        return out

    
class GCNRecon(nn.Module):
    def __init__(self):
        super(GCNRecon, self).__init__()

        self.BasicUnit_ir_en1 = BasicUnit(64)
        self.BasicUnit_ir_en2 = BasicUnit(64)
        self.BasicUnit_ir_en3 = BasicUnit(64)
        self.BasicUnit_vis_en1 = BasicUnit(64)
        self.BasicUnit_vis_en2 = BasicUnit(64)
        self.BasicUnit_vis_en3 = BasicUnit(64)
        self.BasicUnit_de1 = BasicUnit(64)
        self.BasicUnit_de2 = BasicUnit(64)
        self.BasicUnit_de3 = BasicUnit(64)
        # self.fuse_scheme = fuse_scheme # MAX, MEAN, SUM
        self.ir_conv1 =  ConvBlock(128, 64)
        self.vis_conv1 = ConvBlock(128, 64)
        self.ir_conv2 =  ConvBlock(64, 64)
        self.vis_conv2 = ConvBlock(64, 64)
        self.de_conv1 =  ConvBlock(64, 64)
        self.de_conv2 =  ConvBlock(64, 3)

    def forward(self, ir,vis):
        ##########  IR  ######################################
        ir_feature = self.ir_conv1(ir)
        ir_feature = self.ir_conv2(ir_feature)
        ir_feature = self.BasicUnit_ir_en1(ir_feature)
        ir_feature = self.BasicUnit_ir_en2(ir_feature)
        ir_feature = self.BasicUnit_ir_en3(ir_feature)
        ##########  VIS ######################################
        vis_feature = self.vis_conv1(vis)
        vis_feature = self.vis_conv2(vis_feature)
        vis_feature = self.BasicUnit_vis_en1(vis_feature)
        vis_feature = self.BasicUnit_vis_en2(vis_feature)
        vis_feature = self.BasicUnit_ir_en3(vis_feature)
        ##########  Decoder ###################################
        fus_feature = L1_norm(ir_feature, vis_feature)
        fus_feature = self.BasicUnit_de1(fus_feature)# 64 -> 64
        fus_feature = self.BasicUnit_de2(fus_feature)# 64 -> 64
        fus_feature = self.BasicUnit_de3(fus_feature)# 64 -> 64
        fus_feature = self.de_conv1(fus_feature)# 64 -> 3
        out = self.de_conv2(fus_feature)# 64 -> 3
        return out

