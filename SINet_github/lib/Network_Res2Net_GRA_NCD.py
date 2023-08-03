import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.Res2Net_v1b import res2net50_v1b_26w_4s

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x

class RFB_modified(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB_modified, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 3), padding=(0, 1)),
            BasicConv2d(out_channel, out_channel, kernel_size=(3, 1), padding=(1, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 5), padding=(0, 2)),
            BasicConv2d(out_channel, out_channel, kernel_size=(5, 1), padding=(2, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channel, out_channel, 1),
            BasicConv2d(out_channel, out_channel, kernel_size=(1, 7), padding=(0, 3)),
            BasicConv2d(out_channel, out_channel, kernel_size=(7, 1), padding=(3, 0)),
            BasicConv2d(out_channel, out_channel, 3, padding=7, dilation=7)
        )
        self.conv_cat = BasicConv2d(4*out_channel, out_channel, 3, padding=1)
        self.conv_res = BasicConv2d(in_channel, out_channel, 1)

    def forward(self, x):
        x0 = self.branch0(x) 
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        x_cat = self.conv_cat(torch.cat((x0, x1, x2, x3), 1))

        x = self.relu(x_cat + self.conv_res(x))
        return x

class NeighborConnectionDecoder(nn.Module):
    # dense aggregation, it can be replaced by other aggregation previous, such as DSS, amulet, and so on.
    # used after MSF
    def __init__(self, channel):
        super(NeighborConnectionDecoder, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_upsample1 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample2 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample3 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample4 = BasicConv2d(channel, channel, 3, padding=1)
        self.conv_upsample5 = BasicConv2d(2*channel, 2*channel, 3, padding=1)

        self.conv_concat2 = BasicConv2d(2*channel, 2*channel, 3, padding=1)
        self.conv_concat3 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv4 = BasicConv2d(3*channel, 3*channel, 3, padding=1)
        self.conv5 = nn.Conv2d(3*channel, 1, 1)

    def forward(self, x1, x2, x3): 
        x1_1 = x1
        x2_1 = self.conv_upsample1(self.upsample(x1)) * x2 # bs, 32, 16, 16
        x3_1 = self.conv_upsample2(self.upsample(x2_1)) * self.conv_upsample3(self.upsample(x2)) * x3 # bs, 32, 32， 32

        x2_2 = torch.cat((x2_1, self.conv_upsample4(self.upsample(x1_1))), 1) # bs, 64, 16, 16
        x2_2 = self.conv_concat2(x2_2) # bs, 64, 16, 16

        x3_2 = torch.cat((x3_1, self.conv_upsample5(self.upsample(x2_2))), 1) # bs, 96, 32, 32
        x3_2 = self.conv_concat3(x3_2) # bs, 96, 32, 32

        x = self.conv4(x3_2)
        x = self.conv5(x)

        return x

# Group-Reversal Attention (GRA) Block
class GRA(nn.Module):
    def __init__(self, channel, subchannel):
        super(GRA, self).__init__()
        self.group = channel//subchannel 
        self.conv = nn.Sequential(
            nn.Conv2d(channel + self.group, channel, 3, padding=1), nn.ReLU(True),
        )
        self.score = nn.Conv2d(channel, 1, 3, padding=1)

    def forward(self, x, y):
        if self.group == 1:
            x_cat = torch.cat((x, y), 1)
        elif self.group == 2:
            xs = torch.chunk(x, 2, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y), 1)
        elif self.group == 4:
            xs = torch.chunk(x, 4, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y), 1)
        elif self.group == 8:
            xs = torch.chunk(x, 8, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y), 1)
        elif self.group == 16:
            xs = torch.chunk(x, 16, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y), 1)
        elif self.group == 32:
            xs = torch.chunk(x, 32, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y), 1)
        else:
            xs = torch.chunk(x, 64, dim=1)
            x_cat = torch.cat((xs[0], y, xs[1], y, xs[2], y, xs[3], y, xs[4], y, xs[5], y, xs[6], y, xs[7], y,
            xs[8], y, xs[9], y, xs[10], y, xs[11], y, xs[12], y, xs[13], y, xs[14], y, xs[15], y,
            xs[16], y, xs[17], y, xs[18], y, xs[19], y, xs[20], y, xs[21], y, xs[22], y, xs[23], y,
            xs[24], y, xs[25], y, xs[26], y, xs[27], y, xs[28], y, xs[29], y, xs[30], y, xs[31], y,
            xs[32], y, xs[33], y, xs[34], y, xs[35], y, xs[36], y, xs[37], y, xs[38], y, xs[39], y,
            xs[40], y, xs[41], y, xs[42], y, xs[43], y, xs[44], y, xs[45], y, xs[46], y, xs[47], y,
            xs[48], y, xs[49], y, xs[50], y, xs[51], y, xs[52], y, xs[53], y, xs[54], y, xs[55], y,
            xs[56], y, xs[57], y, xs[58], y, xs[59], y, xs[60], y, xs[61], y, xs[62], y, xs[63], y), 1)

        x = x + self.conv(x_cat)
        y = y + self.score(x)

        return x, y

class ReverseStage(nn.Module):
    def __init__(self, channel, ratio):
        super(ReverseStage, self).__init__()
        if ratio > 0:
            in_channel = int(channel*(1+ratio))
            self.first_conv = nn.Conv2d(in_channel, channel,
                                kernel_size=3, padding=1)
        self.weak_gra = GRA(channel, channel)
        self.medium_gra = GRA(channel, 8)
        self.strong_gra = GRA(channel, 1)
        self.ratio = ratio

    def forward(self, x, y):
        # reverse guided block
        y = -1 * (torch.sigmoid(y)) + 1

        # three group-reversal attention blocks
        if self.ratio > 0:
            x = self.first_conv(x)
        x, y = self.weak_gra(x, y)
        x, y = self.medium_gra(x, y)
        _, y = self.strong_gra(x, y)

        return y

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
        x = nn_Unfold(img) 
        x= x.view(B,C,3,3,-1).permute(0,1,4,2,3) 
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

class Network(nn.Module):
    # res2net based encoder decoder
    def __init__(self, channel=32, mode = "qulv",  ratio_list = [0.75,0.75,1], imagenet_pretrained=False):
        super(Network, self).__init__()
        # ---- Backbone ----
        self.backbone = res2net50_v1b_26w_4s(pretrained=imagenet_pretrained)
        channel_lst = [512,1024,2048]
        # ---- Receptive Field Block like module ----
        self.rfb2_1 = RFB_modified(channel_lst[0], channel)
        self.rfb3_1 = RFB_modified(channel_lst[1], channel)
        self.rfb4_1 = RFB_modified(channel_lst[2], channel)
        # ---- Partial Decoder ----
        self.NCD = NeighborConnectionDecoder(channel)

        if mode == "curvature":
            self.cnn_select = CNN_qulv()
        elif mode == 'entropy':
            self.cnn_select = CNN_Entropy()
        else:
            ratio_list =[0,0,0]
        self.ratio_list = ratio_list

        self.ratio_1 = ratio_list[0]
        self.ratio_2 = ratio_list[1]
        self.ratio_3 = ratio_list[2]

        # # ---- reverse stage ----
        self.RS5 = ReverseStage(channel, self.ratio_3)
        self.RS4 = ReverseStage(channel, self.ratio_2)
        self.RS3 = ReverseStage(channel, self.ratio_1)

    def forward(self, x):
        # Feature Extraction
        x_lst = self.backbone(x)
        x2, x3, x4 = x_lst[1], x_lst[2], x_lst[3]

        # Receptive Field Block (enhanced)
        x2_rfb = self.rfb2_1(x2)        # channel -> 32, p13:bs, 32, 32, 32
        x3_rfb = self.rfb3_1(x3)        # channel -> 32, p14:bs, 32, 16, 16
        x4_rfb = self.rfb4_1(x4)        # channel -> 32, p14:bs, 32, 8, 8

        if self.ratio_1 > 0:
            x2_rfb_e = self.cnn_select(x2_rfb, self.ratio_1)  # channel -> 16, p13:bs, 24, 48, 48
            x2_rfb_e = torch.cat((x2_rfb_e, x2_rfb), 1)  # channel -> 16, p13:bs, 24, 48, 48
        else:
            x2_rfb_e = x2_rfb
        
        if self.ratio_2 > 0:
            x3_rfb_e = self.cnn_select(x3_rfb, self.ratio_2)  # channel -> 16, p13:bs, 24, 24, 24
            x3_rfb_e = torch.cat((x3_rfb_e, x3_rfb), 1)  # channel -> 16, p13:bs, 24, 24, 24
        else:
            x3_rfb_e = x3_rfb

        if self.ratio_3 > 0:
            x4_rfb_e = self.cnn_select(x4_rfb, self.ratio_3)  # channel -> 16, p13:bs, 16, 12, 12        
            x4_rfb_e = torch.cat((x4_rfb_e, x4_rfb), 1)  # channel -> 16, p13:bs, 16, 12, 12
        else:
            x4_rfb_e = x4_rfb

        # Neighbourhood Connected Decoder
        S_g = self.NCD(x4_rfb, x3_rfb, x2_rfb)
        S_g_pred = F.interpolate(S_g, scale_factor=8, mode='bilinear')    # Sup-1 (bs, 1, 32, 32) -> (bs, 1, 256, 256)

        # ---- reverse stage 5 ----
        guidance_g = F.interpolate(S_g, scale_factor=0.25, mode='bilinear') #(bs, 1, 32, 32) -> (bs, 1, 8, 8)
        ra4_feat = self.RS5(x4_rfb_e, guidance_g) #(bs, 1, 8, 8)
        S_5 = ra4_feat + guidance_g #(bs, 1, 8, 8)
        S_5_pred = F.interpolate(S_5, scale_factor=32, mode='bilinear')  # Sup-2 (bs, 1, 8, 8) -> (bs, 1, 256, 256)

        # ---- reverse stage 4 ----
        guidance_5 = F.interpolate(S_5, scale_factor=2, mode='bilinear') #(bs, 1, 8, 8) -> (bs, 1, 16, 16)
        ra3_feat = self.RS4(x3_rfb_e, guidance_5) #(bs, 1, 16, 16)
        S_4 = ra3_feat + guidance_5 #(bs, 1, 16, 16)
        S_4_pred = F.interpolate(S_4, scale_factor=16, mode='bilinear')  # Sup-3 (bs, 1, 16, 16) -> (bs, 1, 256, 256)

        # ---- reverse stage 3 ----
        guidance_4 = F.interpolate(S_4, scale_factor=2, mode='bilinear') #(bs, 1, 16, 16) -> (bs, 1, 32, 32)
        ra2_feat = self.RS3(x2_rfb_e, guidance_4) #(bs, 1, 32, 32)
        S_3 = ra2_feat + guidance_4 #(bs, 1, 32, 32)
        S_3_pred = F.interpolate(S_3, scale_factor=8, mode='bilinear')   # Sup-4 (bs, 1, 32, 32) -> (bs, 1, 256, 256)

        return S_g_pred, S_5_pred, S_4_pred, S_3_pred

