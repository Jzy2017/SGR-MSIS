import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

class spatialGCN(torch.nn.Module):
    def __init__(self,in_channels=64):
        super().__init__()
        self.in_channels=in_channels
        self.channels = in_channels // 2
        self.conv1=nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels , kernel_size=1, padding=0)
        self.conv2=nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels , kernel_size=1, padding=0)
        self.conv3=nn.Conv2d(in_channels=self.in_channels, out_channels=self.channels , kernel_size=1, padding=0)
        self.conv4=nn.Conv2d(in_channels=self.channels, out_channels=self.in_channels , kernel_size=1, padding=0)
    def forward(self, input_tensor):
        inputs_shape = input_tensor.size() # (1, 72, 101, 101)
        
        # theta = F.conv2d(input_tensor, torch.randn(channels, in_channels, 1, 1)) # (1, 36, 101, 101)
        theta = self.conv1(input_tensor)
        
        theta = theta.reshape(-1, self.channels, inputs_shape[2] * inputs_shape[3]) # (1, 36, 10201)

        # nu = F.conv2d(input_tensor, torch.randn(self.channels, self.in_channels, 1, 1)) # (1, 36, 101, 101)
        nu = self.conv2(input_tensor)
        nu = nu.reshape(-1, self.channels, inputs_shape[2] * inputs_shape[3]) # (1, 36, 10201)

        nu_tmp = nu.reshape(-1, nu.size()[1] * nu.size()[2]) # (1, 367236)
        nu_tmp = F.softmax(nu_tmp, dim=-1) # (1, 367236)
        nu = nu_tmp.reshape(-1, nu.size()[1], nu.size()[2]) # (1, 36, 10201)
        
        # xi = F.conv2d(input_tensor, torch.randn(self.channels, self.in_channels, 1, 1))
        xi = self.conv3(input_tensor)
        xi = xi.reshape(-1, self.channels, inputs_shape[2] * inputs_shape[3])
        xi_tmp = xi.reshape(-1, xi.size()[1] * xi.size()[2])
        xi_tmp = F.softmax(xi_tmp, dim=-1)
        xi = xi_tmp.reshape(-1, xi.size()[1], xi.size()[2]) # (1, 36, 10201)

        # F_s = torch.matmul(nu, xi.transpose(1, 2)) # (1, 36, 36)
        
        F_s = torch.matmul(nu, xi.transpose(1, 2))
    
        # AF_s = torch.matmul(theta.transpose(1, 2), F_s.transpose(1, 2)) # (1, 10201 ,36)
        AF_s = torch.matmul(theta.transpose(1, 2), F_s.transpose(1, 2)) # (1, 10201 ,36)
        # AF_s = AF_s.transpose(1, 2) # (1, 36, 10201)
        
        AF_s = AF_s.reshape(-1, self.channels, inputs_shape[2], inputs_shape[3]) # (1, 36, 101, 101)
        
        # F_sGCN = F.conv2d(AF_s, torch.randn(self.in_channels, self.channels, 1, 1)) # (1, 72, 101, 101)
        F_sGCN = self.conv4(AF_s)

        return F_sGCN + input_tensor # (1, 72, 101, 101)

class channelGCN(torch.nn.Module):
    def __init__(self,in_channels=64):
        super().__init__()
        self.input_chancel=in_channels
        self.channels = in_channels // 2
        self.C = self.input_chancel // 2
        self.N = self.input_chancel // 4
        self.conv1=nn.Conv2d(in_channels=self.input_chancel, out_channels=self.C , kernel_size=1, padding=0)
        self.conv2=nn.Conv2d(in_channels=self.input_chancel, out_channels=self.N , kernel_size=1, padding=0)
        self.conv3=nn.Conv2d(in_channels=1, out_channels=1 , kernel_size=1, padding=0)
        self.conv4=nn.Conv2d(in_channels=self.N, out_channels=self.N , kernel_size=1, padding=0)

        self.conv5=nn.Conv2d(in_channels=self.N, out_channels=self.input_chancel , kernel_size=1, padding=0)
        self.relu=nn.ReLU(True)
    def forward(self, input_tensor):
        input_shape = input_tensor.size()
        input_chancel = input_shape[-3] # 72
        
        
        # zeta = nn.Conv2d(input_chancel, self.C, kernel_size=1, stride=1, padding=0)(input_tensor) # (1, 36, 101, 101)
        zeta = self.conv1(input_tensor) # (1, 36, 101, 101)
        zeta = zeta.view(input_shape[0], self.C, -1) # (1, 36, 10201)

        # kappa = nn.Conv2d(input_chancel, self.N, kernel_size=1, stride=1, padding=0)(input_tensor)
        kappa = self.conv2(input_tensor)
        kappa = kappa.view(input_shape[0], self.N, -1) # (1, 18, 10201)
        kappa = kappa.permute(0, 2, 1) # (1, 10201, 18)
        
        F_c = torch.matmul(zeta, kappa) # (1, 36, 18)
        
        F_c_tmp = F_c.view(-1, self.C * self.N) # (1, 648)
        F_c_tmp = F.softmax(F_c_tmp, dim=-1)
        F_c = F_c_tmp.view(-1, self.C, self.N) # (1, 36, 18)
        
        F_c = F_c.unsqueeze(1) # (1, 1, 36, 18)
        # F_c = F_c + nn.Conv2d(1, 1, kernel_size=1, padding=0)(F_c) # (1, 1, 36, 18)
        F_c = F_c +self.conv3(F_c) # (1, 1, 36, 18)
        # F_c = nn.functional.relu(F_c)
        F_c = self.relu(F_c)
        F_c = F_c.permute(0, 3, 1, 2) # (1, 18, 1, 36)

        # F_c = nn.Conv2d(self.N, self.N, kernel_size=1, stride=1, padding=0)(F_c) # (1, 18, 1, 36)
        F_c = self.conv4(F_c) # (1, 18, 1, 36)
        F_c = F_c.view(input_shape[0], self.N, self.C) # (1, 18, 36)
        # F_c = torch.matmul(F_c, zeta) # (1, 18, 10201)

        # time.sleep(1000)
        F_c = torch.matmul(zeta.transpose(1, 2), F_c.transpose(1, 2)) # (1, 18, 10201)
        # print(F_c.shape)
        # time.sleep(1000)
        F_c = F_c.unsqueeze(1)
        F_c = F_c.view(input_shape[0], self.N, input_shape[2], input_shape[3]) # (1, 18, 101, 101)
        # F_cGCN = nn.Conv2d(self.N, input_chancel, kernel_size=1, stride=1, padding=0)(F_c) # (1, 72, 101, 101)
        F_cGCN = self.conv5(F_c) # (1, 72, 101, 101)
        
        return F_cGCN + input_tensor # (1, 72, 101, 101)




class BasicUnit(torch.nn.Module):
    def __init__(self,channels=64):
        super().__init__()
        self.channel=channels
        self.sGCN=spatialGCN(channels)
        self.cGCN=channelGCN(channels)
        self.conv1=nn.Conv2d(in_channels=self.channel, out_channels=self.channel , kernel_size=3, dilation=1, padding=1)
        self.conv2=nn.Conv2d(in_channels=self.channel, out_channels=self.channel , kernel_size=3, dilation=1, padding=1)
        self.conv3=nn.Conv2d(in_channels=self.channel, out_channels=self.channel , kernel_size=3, dilation=3, padding=3)
        self.conv4=nn.Conv2d(in_channels=self.channel, out_channels=self.channel , kernel_size=3, dilation=3, padding=3)
        self.conv5=nn.Conv2d(in_channels=5*self.channel, out_channels=self.channel , kernel_size=1, padding=0)
        self.relu=nn.ReLU(True)
    def forward(self, input_tensor):
        # print("hahaha")
        # print(input_tensor.shape)
        channels = input_tensor.shape[-3]
        # F_sGCN = spatialGCN(input_tensor)
        F_sGCN = self.sGCN(input_tensor)
        
        # conv1 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        conv1_out = self.conv1(F.relu(F_sGCN))
        
        # conv2 = nn.Conv2d(channels, channels, kernel_size=3, dilation=1, padding=1)
        conv2_out = self.conv2(F.relu(conv1_out))
        
        # conv3 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        conv3_out = self.conv3(F.relu(F_sGCN))
        
        # conv4 = nn.Conv2d(channels, channels, kernel_size=3, dilation=3, padding=3)
        conv4_out = self.conv4(F.relu(conv3_out))
        
        tmp = torch.cat([F_sGCN, conv1_out, conv2_out, conv3_out, conv4_out], dim=1) # (1, 360, 101, 101)
        
        # F_DCM = nn.Conv2d(5*channels, channels, kernel_size=1, padding=0)
        F_DCM =self.conv5(tmp)
        F_DCM_out = self.relu(F_DCM) # (1, 72, 101, 101)
        
        F_cGCN = self.cGCN(F_DCM_out)
        
        F_unit = F_cGCN + input_tensor
        return  F_unit


def Inference(images, channels=72):
    inchannels = images.size()[-3]

    basic_fea0 = nn.Conv2d(inchannels, channels, kernel_size=3, padding=1)(images)
    basic_fea0 = nn.ReLU(inplace=True)(basic_fea0)
    basic_fea1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)(basic_fea0)
    basic_fea1 = nn.ReLU(inplace=True)(basic_fea1) # (1, 72, 101, 101)

    encode0 = BasicUnit(basic_fea1)
    encode1 = BasicUnit(encode0)
    encode2 = BasicUnit(encode1)
    encode3 = BasicUnit(encode2)
    encode4 = BasicUnit(encode3) # (1, 72, 101, 101)

    middle_layer = BasicUnit(encode4) # (1, 72, 101, 101)

    decoder4 = torch.cat((middle_layer, encode4), dim=-3)
    decoder4 = nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)(decoder4)
    decoder4 = BasicUnit(decoder4)

    decoder3 = torch.cat((decoder4, encode3), dim=-3)
    decoder3 = nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)(decoder3)
    decoder3 = BasicUnit(decoder3)

    decoder2 = torch.cat((decoder3, encode2), dim=-3)
    decoder2 = nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)(decoder2)
    decoder2 = BasicUnit(decoder2)

    decoder1 = torch.cat((decoder2, encode1), dim=-3)
    decoder1 = nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)(decoder1)
    decoder1 = BasicUnit(decoder1)

    decoder0 = torch.cat((decoder1, encode0), dim=-3)
    decoder0 = nn.Conv2d(2*channels, channels, kernel_size=1, padding=0)(decoder0)
    decoder0 = BasicUnit(decoder0)

    decoding_end = torch.cat((decoder0, basic_fea1), dim=-3)
    decoding_end = nn.Conv2d(2*channels, channels, kernel_size=3, padding=1)(decoding_end)
    decoding_end = nn.ReLU(inplace=True)(decoding_end)

    decoding_end = decoding_end + basic_fea0
    res = nn.Conv2d(channels, inchannels, kernel_size=3, padding=1)(decoding_end)
    output = images + res # (1, 3, 101, 101)

    return output

# def Inference(images, channels = 72):
    
    inchannels = images.size()[-1] 
    print(inchannels)
    with torch.variable_scope('Generator', reuse=torch.AUTO_REUSE):       
         
        with torch.variable_scope('basic'):                     
            basic_fea0 = nn.Conv2d(inchannels, channels, kernel_size=3, padding="SAME")(images)
            basic_fea0 = nn.ReLU()(basic_fea0)
            basic_fea1 = nn.Conv2d(channels, channels, kernel_size=3, padding="SAME")(basic_fea0)
            basic_fea1 = nn.ReLU()(basic_fea1)
            
        with torch.variable_scope('encoder0'):         
            encode0 =  BasicUnit(basic_fea1) 
            
        with torch.variable_scope('encoder1'):              
            encode1 =  BasicUnit(encode0) 
        
        with torch.variable_scope('encoder2'): 
            encode2 =  BasicUnit(encode1) 
            
        with torch.variable_scope('encoder3'): 
            encode3 =  BasicUnit(encode2)        

        with torch.variable_scope('encoder4'): 
            encode4 =  BasicUnit(encode3)      

        with torch.variable_scope('middle'):         
            middle_layer = BasicUnit(encode4)

        with torch.variable_scope('decoder4'):              
            decoder4 = torch.cat([middle_layer, encode4], dim=-1)
            decoder4 = nn.Conv2d(2*channels, channels, kernel_size=1, padding="SAME")(decoder4)  
            decoder4 = BasicUnit(decoder4)
            
        with torch.variable_scope('decoder3'):              
            decoder3 = torch.cat([decoder4, encode3], dim=-1)
            decoder3 = nn.Conv2d(2*channels, channels, kernel_size=1, padding="SAME")(decoder3)
            decoder3 = BasicUnit(decoder3)
                                 
        with torch.variable_scope('decoder2'):              
            decoder2 = torch.cat([decoder3, encode2], dim=-1)
            decoder2 = nn.Conv2d(2*channels, channels, kernel_size=1, padding="SAME")(decoder2)  
            decoder2 = BasicUnit(decoder2)
             
        with torch.variable_scope('decoder1'):              
            decoder1 = torch.cat([decoder2, encode1], dim=-1)         
            decoder1 = nn.Conv2d(2*channels, channels, kernel_size=1, padding="SAME")(decoder1)
            decoder1 = BasicUnit(decoder1)
             
        with torch.variable_scope('decoder0'):              
            decoder0 = torch.cat([decoder1, encode0], dim=-1) 
            decoder0 = nn.Conv2d(2*channels, channels, kernel_size=1, padding="SAME")(decoder0)
            decoder0 = BasicUnit(decoder0) 

        with torch.variable_scope('reconstruct'):      
            decoding_end = torch.cat([decoder0, basic_fea1], dim=-1) 
            decoding_end = nn.Conv2d(2*channels, channels, kernel_size=3, padding="SAME")(decoding_end)
            decoding_end = nn.ReLU()(decoding_end)                
                          
            decoding_end = decoding_end + basic_fea0
            res = nn.Conv2d(channels, inchannels, kernel_size=3, padding="SAME")
            output = images + res
    
    return output