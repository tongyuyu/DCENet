
import torch
import torch.nn as nn
import torch.nn.functional as F
from Res import resnet50
# from Swin import Swintransformer
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
Act = nn.ReLU
from utils.non_local_embedded_gaussian import NONLocalBlock2D



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

class SAB(nn.Module):
    def __init__(self, channel):
        super(SAB, self).__init__()
        self.self_att_old=NONLocalBlock2D(in_channels=channel)
        self.self_att_new=NONLocalBlock2D(in_channels=channel)

        self.splayer = nn.Sequential(
                                    nn.Conv2d( channel, 1, 1),
                                    nn.BatchNorm2d(1)
                                    )
        self.chlayer=nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel , bias=False),
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self,x):

        b, c, _, _ = x.size()

        att_ch=self.fc(self.chlayer(x).view(b,c)).view(b, c, 1, 1)
        att_sp=self.splayer(x**2)
        out=att_ch*(att_sp*x)+x
        return out


class Grafting(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=True, qk_scale=None):  #dim=64, num_heads=8, #qkv_bias=True,在生成qkv时是否使用偏置，默认否
        super().__init__()
        self.num_heads = num_heads      #多头注意力中head的个数
        head_dim = dim // num_heads     #dim: 输入token的dim  #计算每一个head需要传入的dim
        self.scale = qk_scale or head_dim ** -0.5   #head_dim的-0.5次方，即1/根号d_k，即理论公式里的分母根号d_k
        self.k = nn.Linear(dim, dim, bias=qkv_bias) #qkv是通过1个全连接层参数为dim和3dim进行初始化的，也可以使用3个全连接层参数为dim和dim进行初始化，二者没有区别
        self.qv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim) #再定义一个全连接层，是 将每一个head的结果进行拼接的时候乘的那个矩阵W^O
        self.act = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(8, 8, kernel_size=3, stride=1, padding=1)
        # self.lnx = nn.LayerNorm(64)
        # self.lny = nn.LayerNorm(64)
        self.convx = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.convy = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.bn = nn.BatchNorm2d(8)
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )


    def forward(self, x, y):
        batch_size = x.shape[0] # 8
        chanel = x.shape[1]     # 64
        sc = x                  # 8 64 28 28
        x = self.convx(x) # 8 64 28 28
        y = self.convy(y) # 8 64 28 28
        x = x.view(batch_size, chanel, -1).permute(0, 2, 1) # 8 784 64
        sc1 = x  # 8 784 64

        y = y.view(batch_size, chanel, -1).permute(0, 2, 1) # 8 784 64


        B, N, C = x.shape # 8 784 64
        y_k = self.k (y).reshape(B, N, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)   # 1 8 8 784 8
        x_qv = self.qv(x).reshape(B, N, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # 2 8 8 784 8
        x_q, x_v = x_qv[0], x_qv[1]     # 8 8 784 8
        y_k = y_k[0] # 8 8 784 8
        attn = (x_q @ y_k.transpose(-2, -1)) * self.scale   # 8 8 784 784
        # 现在的操作都是对每个head进行操作
        # transpose是转置最后2个维度，@就是矩阵乘法的意思
        # q  [batchsize, num_heads, num_patches+1, embed_dim_per_head]
        # k^T[batchsize, num_heads, embed_dim_per_head, num_patches+1]
        # q*k^T=[batchsize, num_heads, num_patches+1, num_patches+1]
        # self.scale=head_dim的-0.5次方
        # 至此完成了(Q*K^T)/根号d_k的操作
        attn = attn.softmax(dim=-1)  # 8 8 784 784
        # dim=-1表示在得到的结果的每一行上进行softmax处理，-1就是最后1个维度
        # 至此完成了softmax[(Q*K^T)/根号d_k]的操作

        x = (attn @ x_v).transpose(1, 2).reshape(B, N, C) # 8 784 64
        # @->[batchsize, num_heads, num_patches+1, embed_dim_per_head]
        # 这一步矩阵乘积就是加权求和
        # transpose->[batchsize, num_patches+1, num_heads, embed_dim_per_head]
        # reshape->[batchsize, num_patches+1, num_heads*embed_dim_per_head]即[batchsize, num_patches+1, total_embed_dim]
        # reshape实际上就实现了concat拼接

        x = self.proj(x)  # 8 784 64
        #将上一步concat的结果通过1个线性映射，通常叫做W，此处用全连接层实现
        x = (x + sc1)

        x = x.permute(0, 2, 1)# 8 64 784
        x = x.view(batch_size, chanel, *sc.size()[2:]) # 8 64 28 28
        x = self.conv2(x) + x
        return x

class Trans(nn.Module):
    def __init__(self, in_channel, out_channel) -> None:
        super(Trans, self).__init__()
        # self.sqz_s = nn.Sequential(
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True)
        # )
        # self.sqz_r = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True)
        # )

        self.GF = Grafting(64, num_heads=8)


    def forward(self, r,s):
        #s2 = F.interpolate(s, size=(28,28), mode='bilinear', align_corners=True)
        #r = F.interpolate(r, size=(28,28), mode='bilinear', align_corners=True)

        # s3 = self.sqz_s(s2)
        # r2 = self.sqz_r(r)
        rs = self.GF(r, s)  # 8 64 28 28

        return rs

    def initialize(self):
        weight_init(self)
class multi_head_attenionLayer(nn.Module):
    def __init__(self, channel):
        super(multi_head_attenionLayer, self).__init__()

        self.convTo2 = nn.Conv2d(channel * 2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.SAB = SAB(channel=64)
        self.Trans = Trans(64,64)

    def forward(self,x_r, x_s):


        H = torch.cat((x_r,x_s), dim=1)
        H_conv = self.convTo2(H)
        H_conv = self.sig(H_conv)
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        x_r = x_r * ga
        x_s = x_s * gm

        x_s_1 = self.SAB(x_s)

        x_out = self.Trans(x_r ,x_s_1)


        return x_out


class ALC(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ALC, self).__init__()
        # self.sqz_H = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True)
        # )
        # self.sqz_L = nn.Sequential(
        #     nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=1, dilation=1, padding=1),
        #     nn.BatchNorm2d(out_channel),
        #     nn.ReLU(inplace=True)
        # )
        self.conv1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.conv_cat = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, fL, fH):
        # fL = self.sqz_L(fL)
        # fH = self.sqz_H(fH)
        fL_H = F.interpolate(fL, size=fH.size()[2:], mode='bilinear', align_corners=True)
        fH_L = F.interpolate(fH, size=fL.size()[2:], mode='bilinear', align_corners=True)

        fH_1 = self.conv1(fH)
        fL_H_1 = self.conv2(fL_H)
        fH_L_1 = self.conv3(fH_L)
        fL_1 = self.conv4(fL)

        fL_H_2 = fH_1.mul(fL_H_1)
        fH_L_2 = fL_1.mul(fH_L_1)
        fL_H_3 = fL_H_2 + fH_1
        fH_L_3 = fH_L_2 + fL_1

        fL_H_4 = self.conv5(fL_H_3)
        fH_L_4 = self.conv6(fH_L_3)

        fL_H_5 = F.interpolate(fL_H_4, size=fH_L_4.size()[2:], mode='bilinear', align_corners=True)

        # print('fL_H_5.size',fL_H_5.size())
        # print('fH_L_4.size',fH_L_4.size())

        fc = torch.cat((fH_L_4, fL_H_5), 1)
        fout = self.conv_cat(fc)

        return fout




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


        self.sqz_r5 = BasicConv2d(2048, 64, 1)
        self.sqz_r4 = BasicConv2d(1024, 64, 1)
        self.sqz_r3 = BasicConv2d(512, 64, 1)
        self.sqz_r2 = BasicConv2d(256, 64, 1)

        self.sqz_s4 = BasicConv2d(512, 64, 1)
        self.sqz_s3 = BasicConv2d(512, 64, 1)
        self.sqz_s2 = BasicConv2d(256, 64, 1)
        self.sqz_s1 = BasicConv2d(128, 64, 1)

        self.conv448_64 = BasicConv2d(448, 64, 1)
        self.conv256_64 = BasicConv2d(256, 64, 1)

        self.ALC1 = ALC(64,64)
        self.ALC2 = ALC(64,64)
        self.ALC3 = ALC(64,64)
        self.ALC4 = ALC(64,64)
        self.ALC5 = ALC(64,64)
        self.ALC6 = ALC(64,64)

        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.multihead_att_256= multi_head_attenionLayer(64).cuda()

        self.resnet    = resnet50()
        # self.swin      = Swintransformer(224)
        # self.swin.load_state_dict(torch.load('../pre/swin224.pth')['model'],strict=False)
        self.resnet.load_state_dict(torch.load('../pre/resnet50.pth'),strict=False)
        
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

        r2,r3,r4,r5 = self.resnet(x)
        # s1,s2,s3,s4 = self.swin(y)

        # s4_1 = self.sqz_s4(s4) # 64 14 14
        # s3_1 = self.sqz_s3(s3) # 64 14 14
        # s2_1 = self.sqz_s2(s2) # 64 28 28 
        # s1_1 = self.sqz_s1(s1) # 64 56 56
        print('r5.size:',r5.size())   # 2048 33 33
        print('r4.size:',r4.size())   # 1024 66 66
        print('r3.size:',r3.size())   # 512 132 132
        print('r2.size:',r2.size())   # 256 264 265
        r5_1 = self.sqz_r5(r5) # 64 31 31
        r4_1 = self.sqz_r4(r4) # 64 62 62
        r3_1 = self.sqz_r3(r3) # 64 124 124
        r2_1 = self.sqz_r2(r2) # 64 248 248
        print('r5_1.size:',r5_1.size())
        print('r4_1.size:',r4_1.size())
        print('r3_1.size:',r3_1.size())
        print('r2_1.size:',r2_1.size())
        
        r5_1_up = self.upsample2(r5_1)  # 8 64 62 62
        
        r54 = torch.cat((r5_1_up,r4_1),1) # 8 128 62 62
        r54_up = self.upsample2(r54)  # 8 128 124 124
        r43 = torch.cat((r54_up,r3_1),1)
        r43_up = self.upsample2(r43)  # 8 192 248 248
        r32 = torch.cat((r43_up,r2_1),1) # 8 256 248 248
        
        


        # 全是64通道，信息可能有损耗
        r23 = self.conv256_64(r32) # 8 64 248 248

        
        pred1 = F.interpolate(self.linear1(r23), size=shape, mode='bilinear') 
        wr = F.interpolate(self.linear2(r23), size=(28,28), mode='bilinear') 
        ws = F.interpolate(self.linear3(r23), size=(28,28), mode='bilinear') 
        #attmap = self.conv(attmap)
        #print('wr.zise',wr.shape)
        #print('ws.zise',ws.shape)
        #print('attmap.zise',attmap.shape)

        #return pred1,wr,ws,attmap
        return pred1,wr,ws

    

