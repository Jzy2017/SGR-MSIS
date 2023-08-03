import numpy as np
from tensorDLT_function import solve_DLT
from spatial_transform import Transform
from output_spatial_transform import Transform_output
import torch.nn.functional as F
import torch
from output_tensorDLT import output_solve_DLT
import torch.nn as nn
from casgcn_share import GraphReasoning
from gcn import BasicUnit
from feature_output_spatial_transform import Transform_output_feature
class CGR(nn.Module):
    def __init__(self, n_class=2, n_iter=2, chnn_side=(512, 256, 128), chnn_targ=(512, 128, 32, 4), rd_sc=32, dila=(4, 8, 16)):
        super().__init__()
        self.n_graph = len(chnn_side)#3
        n_node = len(dila)
        graph = [GraphReasoning(ii, rd_sc, dila, n_iter) for ii in chnn_side]
        self.graph = nn.ModuleList(graph)
        C_cat = [nn.Sequential(
            nn.Conv2d(ii//rd_sc*n_node, ii//rd_sc, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ii//rd_sc),
            nn.ReLU(inplace=True))
            for ii in (chnn_side+chnn_side)]
        self.C_cat = nn.ModuleList(C_cat)
        idx = [ii for ii in range(len(chnn_side))]
        C_up = [nn.Sequential(
            nn.Conv2d(chnn_targ[ii]+chnn_side[ii]//rd_sc, chnn_targ[ii+1], 3, 1, 1, bias=False),
            nn.BatchNorm2d(chnn_targ[ii+1]),
            nn.ReLU(inplace=True))
            for ii in (idx+idx)]
        self.C_up = nn.ModuleList(C_up)
        self.C_cls = nn.Conv2d(chnn_targ[-1]*2, n_class, 1)

    def forward(self, img,depth):
        # img, depth = inputs
        cas_rgb, cas_dep = img[0], depth[0]# (b,512,h/8,w/8), (b,512,h/8,w/8)
        # cas_rgb, cas_rgb, cas_rgb, cas_rgb, cas_rgb, cas_rgb = inputs
        nd_rgb, nd_dep, nd_key = None, None, False
        for ii in range(self.n_graph):#range(3)
            feat_rgb, feat_dep = self.graph[ii]([img[ii], depth[ii], nd_rgb, nd_dep], nd_key)
            feat_rgb = torch.cat(feat_rgb, 1)# b,48,h/8,w/8
            feat_rgb = self.C_cat[ii](feat_rgb)# b,48,h/8,w/8
            feat_dep = torch.cat(feat_dep, 1)
            feat_dep = self.C_cat[self.n_graph+ii](feat_dep)
            nd_rgb, nd_dep, nd_key = feat_rgb, feat_dep, True
            cas_rgb = torch.cat((feat_rgb, cas_rgb), 1)
            cas_rgb = F.interpolate(cas_rgb, scale_factor=2, mode='bilinear', align_corners=True)
            cas_rgb = self.C_up[ii](cas_rgb)
            cas_dep = torch.cat((feat_dep, cas_dep), 1)
            cas_dep = F.interpolate(cas_dep, scale_factor=2, mode='bilinear', align_corners=True)
            cas_dep = self.C_up[self.n_graph+ii](cas_dep)
        feat = torch.cat((cas_rgb, cas_dep), 1)
        out = self.C_cls(feat)
        return out
		
class feature_extractor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Sequential( nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True),
                                        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU(True)
                                        )
        self.BasicUnit1=BasicUnit(64)
        self.BasicUnit2=BasicUnit(64)
        self.BasicUnit3=BasicUnit(64)
    def forward(self, input):
        conv1 = self.conv1(input.float())
        conv2 =  self.BasicUnit1(conv1) # 72->72             
        conv3 =  self.BasicUnit2(conv2)  # 72->72
        conv4 =  self.BasicUnit3(conv3)  # 72->72
        return conv1,conv2,conv3,conv4
class VGG2(nn.Module):
    def __init__(self):
        super(VGG2, self).__init__()
        # conv1, 2 layers
        self.pool1 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv2, 2 layers
        self.conv2_1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv3, 4 layers
        self.conv3_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_3 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.conv3_4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn3_4 = nn.BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True)
        self.relu3_4 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv4, 4 layers
        self.conv4_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.conv4_4 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn4_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu4_4 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        # conv5, 4 layers
        dila = [2, 4, 8, 16]
        self.conv5_1 = nn.Conv2d(256, 256, 3, padding=dila[0], dilation=dila[0])
        self.bn5_1 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(256, 256, 3, padding=dila[1], dilation=dila[1])
        self.bn5_2 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(256, 256, 3, padding=dila[2], dilation=dila[2])
        self.bn5_3 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.conv5_4 = nn.Conv2d(256, 256, 3, padding=dila[3], dilation=dila[3])
        self.bn5_4 = nn.BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True)
        self.relu5_4 = nn.ReLU(inplace=True)

    def forward(self, x):
        h1 = x  
        h = self.pool1(h1)
        h = self.relu2_1(self.bn2_1(self.conv2_1(h)))
        h = self.relu2_2(self.bn2_2(self.conv2_2(h)))
        h_nopool2 = h
        h = self.pool2(h)
        h2 = h_nopool2              
        h = self.relu3_1(self.bn3_1(self.conv3_1(h)))
        h = self.relu3_2(self.bn3_2(self.conv3_2(h)))
        h = self.relu3_3(self.bn3_3(self.conv3_3(h)))
        h = self.relu3_4(self.bn3_4(self.conv3_4(h)))
        h_nopool3 = h
        h = self.pool3(h)
        h3 = h_nopool3             
        h = self.relu4_1(self.bn4_1(self.conv4_1(h)))
        h = self.relu4_2(self.bn4_2(self.conv4_2(h)))
        h = self.relu4_3(self.bn4_3(self.conv4_3(h)))
        h = self.relu4_4(self.bn4_4(self.conv4_4(h)))
        h_nopool4 = h
        h = self.relu5_1(self.bn5_1(self.conv5_1(h)))
        h = self.relu5_2(self.bn5_2(self.conv5_2(h)))
        h = self.relu5_3(self.bn5_3(self.conv5_3(h)))
        h = self.relu5_4(self.bn5_4(self.conv5_4(h)))
        h5 = h                      
        return h5, h3, h2,h1  #h4 h1 

class VGG1(nn.Module):
    def __init__(self):
        super(VGG1, self).__init__()
        # conv1, 2 layers
        self.conv1_1 = nn.Conv2d(3, 64, 3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
    def forward(self, x):
        h = x  
        h = self.relu1_1(self.conv1_1(h))
        h = self.relu1_2(self.conv1_2(h))
        h_nopool1 = h
        h1 = h_nopool1  
        return h1  


class H_estimator(torch.nn.Module):
    def __init__(self, batch_size, device, is_training=False):
        super().__init__()
        self.device=device
        self.M_tile_inv_128, self.M_tile_128 = self.to_transform_H(128, batch_size)
        self.transform128 = Transform(128,128,self.device,batch_size)
        if device > -1:
            self.transform128=self.transform128.to(self.device)
        self.graph = CGR()    
        self.keep_prob = 0.5 if is_training==True else 1.0
        self.getoffset = torch.nn.Sequential(torch.nn.Linear(in_features = 2*128*128, out_features = 1024),
                                            nn.ReLU(True),
                                            nn.Dropout(p = self.keep_prob),
                                            torch.nn.Linear(in_features =1024, out_features = 8))
        self.transform_output=Transform_output()
        self.transform_output_feature=Transform_output_feature()

    def to_transform_H(self, patch_size, batch_size):            
        M = np.array([[patch_size / 2.0, 0., patch_size / 2.0],
                    [0., patch_size / 2.0, patch_size / 2.0],
                    [0., 0., 1.]]).astype(np.float32)
        M_tensor = torch.from_numpy(M)
        M_tile = torch.unsqueeze(M_tensor, 0).repeat( [batch_size, 1, 1])
        M_inv = np.linalg.inv(M)
        M_tensor_inv = torch.from_numpy(M_inv)
        M_tile_inv = torch.unsqueeze(M_tensor_inv, 0).repeat([batch_size, 1, 1])
        if self.device>-1:
            M_tile_inv = M_tile_inv.to(self.device)
            M_tile=M_tile.to(self.device)
        return M_tile_inv, M_tile

    def forward(self, feature_ir1,feature_ir2, inputs_ir, feature_vis1, feature_vis2, inputs_vis,\
        size, ir1_f0,ir2_f0,vis1_f0,vis2_f0):

        batch_size = inputs_ir.shape[0]
        ############### build_model ###################################
        ir_input1 = inputs_ir[...,0:3].permute(0,3,1,2)
        ir_input2 = inputs_ir[...,3:6].permute(0,3,1,2)
         
        vis_input1 = inputs_vis[...,0:3].permute(0,3,1,2)
        vis_input2 = inputs_vis[...,3:6].permute(0,3,1,2)   
        ##############################  feature_extractor ##############################       
        ir1_f3,  ir1_f2,  ir1_f1, _ = feature_ir1
        ir2_f3,  ir2_f2,  ir2_f1,  _ = feature_ir2
        vis1_f3, vis1_f2, vis1_f1, _ = feature_vis1
        vis2_f3, vis2_f2, vis2_f1, _ = feature_vis2

        mix1_f1 = torch.cat((ir1_f1,vis1_f1),1)
        mix2_f1 = torch.cat((ir2_f1,vis2_f1),1)


        mix1_f2=torch.cat((ir1_f2,vis1_f2),1)
        mix2_f2=torch.cat((ir2_f2,vis2_f2),1)
 

        mix1_f3=torch.cat((ir1_f3,vis1_f3),1)
        mix2_f3=torch.cat((ir2_f3,vis2_f3),1)

        out = self.graph([mix1_f3,mix1_f2,mix1_f1], [mix2_f3,mix2_f2,mix2_f1])
        flatten = out.contiguous().view(out.shape[0],-1)
        
        offset = self.getoffset(flatten)
        offset = torch.unsqueeze(offset, 2)#*128

        size_tmp = torch.cat([size,size,size,size],axis=1)/128.
        offset = torch.mul(offset, size_tmp)
        H_mat = output_solve_DLT(offset, size)  
        irs=inputs_ir.permute(0,3,1,2)
        viss=inputs_vis.permute(0,3,1,2)

        ir_warp = self.transform_output(irs, H_mat,size,offset)
        vis_warp = self.transform_output(viss, H_mat,size,offset)
        ir_feature=torch.cat((ir1_f0,ir2_f0),1)
        vis_feature=torch.cat((vis1_f0,vis2_f0),1)
        ir_f_warp = self.transform_output_feature(ir_feature, H_mat,size,offset)
        vis_f_warp = self.transform_output_feature(vis_feature, H_mat,size,offset)
        return offset, ir_warp, vis_warp, ir_f_warp, vis_f_warp

        

