import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import os
import matplotlib.pyplot as plt
import numpy as np
import heapq

class CNN_Entropy(nn.Module):
    def __init__(self, win_w=3, win_h=3):
        super(CNN_Entropy, self).__init__()
        self.win_w = win_w
        self.win_h = win_h

    def calcIJ_new(self, img_patch):
        total_p = img_patch.shape[-1] * img_patch.shape[-2]
        if total_p % 2 != 0:
            tem = torch.flatten(img_patch, start_dim=-2, end_dim=-1) 
            center_p = tem[:, :, :, int(total_p / 2)]
            mean_p = (torch.sum(tem, dim=-1) - center_p) / (total_p - 1)
            if torch.is_tensor(img_patch):
                return center_p * 100 + mean_p
            else:
                return (center_p, mean_p)
        else:
            print("modify patch size")

    def forward(self, img, ratio):
        B, C, H, W = img.shape
        ext_x = int(self.win_w / 2)
        ext_y = int(self.win_h / 2)

        new_width = ext_x + W + ext_x 
        new_height = ext_y + H + ext_y
        
        nn_Unfold=nn.Unfold(kernel_size=(self.win_w,self.win_h),dilation=1,padding=ext_x,stride=1)
        x = nn_Unfold(img) # (B,C*K*K,L)
        x= x.view(B,C,3,3,-1).permute(0,1,4,2,3) # (B,C*K*K,L) ---> (B,C,L,K,K)
        ij = self.calcIJ_new(x).reshape(B*C, -1) 

        h = []  
        for j in range(ij.shape[0]):
            Fij = torch.unique(ij[j].detach(),return_counts=True,dim=0)[1]
            p = Fij * 1.0 / (new_height * new_width) 
            h_tem = -p * (torch.log(p) / torch.log(torch.as_tensor(2.0)))
            a = torch.sum(h_tem) 
            h.append(a)
        H = torch.stack(h,dim=0).reshape(B,C) 
        
        _, index = torch.topk(H, int(ratio*C), dim=1) # Nx3
        selected = []
        for i in range(img.shape[0]):
            selected.append(torch.index_select(img[i], dim=0, index=index[i]).unsqueeze(0))
        selected = torch.cat(selected, dim=0)
        
        return selected
    
class CNN_qulv(torch.nn.Module):
    def __init__(self):
        super(CNN_qulv, self).__init__()
        weights = torch.tensor([[[[-1/16, 5/16, -1/16], [5/16, -1, 5/16], [-1/16, 5/16, -1/16]]]])
        self.weight = torch.nn.Parameter(weights).cuda() 

    def forward(self, x, ratio):
        x_origin = x
        x = x.reshape(x.shape[0]*x.shape[1],1,x.shape[2],x.shape[3])
        out = F.conv2d(x, self.weight) 
        out = torch.abs(out)
        p = torch.sum(out, dim=-1)
        p = torch.sum(p, dim=-1)
        p=p.reshape(x_origin.shape[0], x_origin.shape[1])

        _, index = torch.topk(p, int(ratio*x_origin.shape[1]), dim=1) # Nx3
        selected = []
        for i in range(x_origin.shape[0]):
            selected.append(torch.index_select(x_origin[i], dim=0, index=index[i]).unsqueeze(0))
        selecte = torch.cat(selected, dim=0)
        return selecte
    
class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm, ratio_list, mode):
        super(Decoder, self).__init__()
        if backbone == 'resnet' or backbone == 'drn':
            low_level_inplanes = 256
        elif backbone == 'xception':
            low_level_inplanes = 128
        elif backbone == 'mobilenet':
            low_level_inplanes = 24
        else:
            raise NotImplementedError
        
        if mode == "curvature":
            self.cnn_select = CNN_qulv()
        elif mode == 'entropy':
            self.cnn_select = CNN_Entropy()
        else:
            ratio_list =[0,0]
        self.ratio_list = ratio_list
        
        self.conv1 = nn.Conv2d(low_level_inplanes, 48, 1, bias=False)
        self.bn1 = BatchNorm(48)
        self.relu = nn.ReLU()
        
        in_channel = int(48*(1 + ratio_list[0]) + 256*(1+ ratio_list[1]))
        self.last_conv = nn.Sequential(nn.Conv2d(in_channel, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))
        
        self._init_weight()


    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        low_level_feat = self.bn1(low_level_feat)
        low_level_feat = self.relu(low_level_feat)

        if self.ratio_list[0] > 0:
            low_level_feat_select = self.cnn_select(low_level_feat, self.ratio_list[0])
            low_level_feat = torch.cat((low_level_feat_select,low_level_feat), dim=1)

        if self.ratio_list[1] > 0:
            x_select = self.cnn_select(x, self.ratio_list[1])
            x = torch.cat((x_select,x), dim=1)

        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.last_conv(x)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_decoder(num_classes, backbone, BatchNorm, ratio_list, mode):
    return Decoder(num_classes, backbone, BatchNorm, ratio_list, mode)