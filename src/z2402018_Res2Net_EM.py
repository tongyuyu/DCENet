
import torch
import torch.nn as nn
import torch.nn.functional as F
from Res2Net_v1b import res2net50_v1b_26w_4s
# from Res import resnet50
# from Swin import Swintransformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
Act = nn.ReLU
from utils.non_local_embedded_gaussian import NONLocalBlock2D



#Global Contextual module
class RFB(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(RFB, self).__init__()
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


    def initialize(self):
        weight_init(self)

def weight_init(module):
    for n, m in module.named_children():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d,nn.BatchNorm1d)):
            nn.init.ones_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Sequential):
            weight_init(m)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.ReLU,Act,nn.AdaptiveAvgPool2d,nn.Softmax)):
            pass
        else:
            m.initialize()

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x=max_out
        x = self.conv1(x)
        return self.sigmoid(x)
    


class BasicConv(nn.Module):
    def __init__(
        self,
        in_planes,
        out_planes,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        relu=True,
        bn=True,
        bias=False,
    ):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )
        self.bn = (
            nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            if bn
            else None
        )
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1
        )


class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(
            2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False
        )

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid_(x_out)
        return x * scale


class MultipleAttention(nn.Module):
    def __init__(
        self,
        gate_channels,
        reduction_ratio=16,
        pool_types=["avg", "max"],
        no_spatial=False,
    ):
        super(MultipleAttention, self).__init__()
        self.ChannelGateH = SpatialGate()
        self.ChannelGateW = SpatialGate()
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
        self.conv4to1 = nn.Conv2d(4*gate_channels, gate_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, x):
        x_perm1 = x.permute(0, 2, 1, 3).contiguous()
        x_out1 = self.ChannelGateH(x_perm1)
        x_out11 = x_out1.permute(0, 2, 1, 3).contiguous()
        x_perm2 = x.permute(0, 3, 2, 1).contiguous()
        x_out2 = self.ChannelGateW(x_perm2)
        x_out21 = x_out2.permute(0, 3, 2, 1).contiguous()

        # Cat
        x_max = torch.max(x_out11,x_out21)
        # Max
        #x_cat = x_cat.max(dim=1)[0]
        x_out = x_max*x

        return x_out




class ALC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ALC, self).__init__()
       
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.downsample = nn.MaxPool2d(3,2,1)
        self.CA = ChannelAttention(64)
        self.SA = SpatialAttention()
        self.GCM = MultipleAttention(64)

    def forward(self, x):
        
        x_m = self.GCM(x)
        
        x_s = self.downsample(x)
        x_s1 = self.CA(x_s)
        # print('x_s1.size():', x_s1.size())
        x_s2 = torch.mul(x_s, x_s1)
        
        x_s3 = self.SA(x_s2)
        x_s4 = torch.mul(x_s2,x_s3)
        x_s4_up = self.upsample2(x_s4)

        # print('x_s4_up.size():', x_s4_up.size())
        # print('x_m.size():', x_m.size())
        x_m1 = torch.mul(x_m, x_s4_up)
        x_out = x + x_m + x_m1 + x_s4_up
        # print('x_out.size():', x_out.size())
        

        return x_out
    
    
    
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 64
        modules = []
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),)

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)

class IntegralAttention (nn.Module):
    def __init__(self, in_channel, out_channel):
        super(IntegralAttention, self).__init__()
        self.relu = nn.ReLU(True)
        self.branch0 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,kernel_size=3, stride=1,padding=1),
        )
        self.branch1 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=3, dilation=3)
        )
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channel, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel, 3, padding=5, dilation=5)
        )
        self.conv_cat = nn.Sequential(
            BasicConv2d(out_channel*3, out_channel,kernel_size=3, stride=1,padding=1),
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1),

        )
        self.conv_res = BasicConv2d(in_channel, out_channel,kernel_size=3, stride=1,padding=1)

        self.eps = 1e-5   
        self.IterAtt = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 8, kernel_size=1),
            nn.LayerNorm([out_channel // 8, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel // 8, out_channel, kernel_size=1)
        )
        self.ConvOut = nn.Sequential(
            BasicConv2d(out_channel, out_channel,kernel_size=3, stride=1,padding=1), 
        )

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        x_cat = self.conv_cat(torch.cat((x0, x1, x2), 1))
        fuse = self.relu(x_cat + self.conv_res(x))

        # can change to the MS-CAM or SE Attention, refer to lib.RCAB.
        context = (fuse.pow(2).sum((2,3), keepdim=True) + self.eps).pow(0.5) # [B, C, 1, 1]
        channel_add_term = self.IterAtt(context)
        out = channel_add_term * fuse + fuse

        out = self.ConvOut(fuse)

        return out


class PGNet(nn.Module):
    def __init__(self, cfg=None):
        super(PGNet, self).__init__()
        self.cfg      = cfg
        self.linear1 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear2 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.linear3 = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        self.conv = nn.Conv2d(8,1,kernel_size=3, stride=1, padding=1)
        
        
        if self.cfg is None or self.cfg.snapshot is None:
            weight_init(self)




        self.conv448_64 = BasicConv2d(448, 64, 1)
        self.conv320_64 = BasicConv2d(320, 64, 1)


        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        
        
        self.attention5 = IntegralAttention(2048, 64)
        self.attention4 = IntegralAttention(1024, 64)
        self.attention3 = IntegralAttention(512, 64)
        self.attention2 = IntegralAttention(256, 64)
        self.attention1 = IntegralAttention(64, 64)


        
        
        
        # self.resnet    = resnet50() 
        self.resnet    = res2net50_v1b_26w_4s()
        # self.swin      = Swintransformer(224)
        # self.swin.load_state_dict(torch.load('../pre/swin224.pth')['model'],strict=False)
        # self.resnet.load_state_dict(torch.load('../pre/resnet50.pth'),strict=False)
        self.resnet.load_state_dict(torch.load('../pre/res2net50_v1b_26w_4s-3cf99910.pth'),strict=False)
        
        if self.cfg is not None and self.cfg.snapshot:
            print('load checkpoint')
            pretrain=torch.load(self.cfg.snapshot)
            new_state_dict = {}
            for k,v in pretrain.items():
                new_state_dict[k[7:]] = v  
            self.load_state_dict(new_state_dict, strict=False)  

    def forward(self, x,shape=None,mask=None):
        shape = x.size()[2:] if shape is None else shape
        # y = F.interpolate(x, size=(224,224), mode='bilinear',align_corners=True)

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)#  64 x 264 264 
        # print('x.size():',x.size())
        # layer1
        r2 = self.resnet.layer1(x)  # 256 x 88 x 88
        # layer2
        r3 = self.resnet.layer2(r2)  # 512 x 44 x 44
        # layer3
        r4 = self.resnet.layer3(r3)  # 1024 x 22 x 22
        # layer 4
        r5 = self.resnet.layer4(r4)  # 2048 x 11 x 11


        # print('r5.size:',r5.size())   # 2048 33 33
        # print('r4.size:',r4.size())   # 1024 66 66
        # print('r3.size:',r3.size())   # 512 132 132
        # print('r2.size:',r2.size())   # 256 264 264

        
        r5 = F.interpolate(r5, size=(62,62), mode='bilinear', align_corners=True) # 8 64 31 31
        r4 = F.interpolate(r4, size=(62,62), mode='bilinear', align_corners=True) # 8 64 62 62
        r3 = F.interpolate(r3, size=(124,124), mode='bilinear', align_corners=True) # 8 64 124 124
        r2 = F.interpolate(r2, size=(248,248), mode='bilinear', align_corners=True) # 8 64 248 248
        r1 = F.interpolate(x, size=(248,248), mode='bilinear', align_corners=True) # 8 64 248 248
        
        x5_1 = self.attention5(r5) # 64 62 62
        x4_1 = self.attention4(r4) # 64 62 62
        x3_1 = self.attention3(r3) # 64 124 124
        x2_1 = self.attention2(r2) # 64 248 248
        x1_1 = self.attention1(r1) # 64 248 248

        x54 = torch.cat((x5_1, x4_1),1)# 8 128 62 62

        x54_up = self.upsample2(x54)  # 8 128 124 124
        x43 = torch.cat((x54_up,x3_1),1)
        x43_up = self.upsample2(x43)  # 8 192 248 248
        x32 = torch.cat((x43_up,x2_1),1) # 8 256 248 248
        x21 = torch.cat((x32,x1_1),1) # 8 320 248 248
        
        


        # 全是64通道，信息可能有损耗
        r23 = self.conv320_64(x21) # 8 64 248 248

        
        pred1 = F.interpolate(self.linear1(r23), size=shape, mode='bilinear') 
        wr = F.interpolate(self.linear2(r23), size=(28,28), mode='bilinear') 
        ws = F.interpolate(self.linear3(r23), size=(28,28), mode='bilinear') 

        return pred1,wr,ws
