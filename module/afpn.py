from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

# from model.module.RFACA_Conv import RFCAConv
# from model.module.refConv import RepConvBlock

class DepthwiseSeparableConv(nn.Module):
    # def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=2, padding=0):
        super(DepthwiseSeparableConv, self).__init__()

        # 深度卷积层
        self.depthwise = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size,
                                                 stride, padding, groups=in_channels, bias=False),
                                       nn.BatchNorm2d(in_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       # nn.LeakyReLU(0.08, inplace=True)
                                       # nn.ReLU(inplace=True)
                                       )
        # 逐点卷积层
        self.pointwise = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                       nn.BatchNorm2d(out_channels),
                                       # activation_layer
                                       nn.LeakyReLU(0.1, inplace=True)
                                       # nn.LeakyReLU(0.1, inplace=True)
                                       # nn.ReLU(inplace=True)
                                       )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x



def Conv(filter_in, filter_out, kernel_size, stride=1, pad=0):
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("silu", nn.SiLU(inplace=True)),
    ]))


class Downsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(Downsample, self).__init__()

        self.downsample = nn.Sequential(
            Conv(in_channels, out_channels, scale_factor, scale_factor, 0)
        )

    def forward(self, x):
        x = self.downsample(x)

        return x





class ASFF_3(nn.Module):
    def __init__(self, inter_dim=512, level=0, channel=[128, 256,512]):
        super(ASFF_3, self).__init__()

        self.inter_dim = inter_dim
        compress_c = 8

        self.weight_level_1 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_2 = Conv(self.inter_dim, compress_c, 1, 1)
        self.weight_level_3 = Conv(self.inter_dim, compress_c, 1, 1)

        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)

        # self.conv = Conv(self.inter_dim, self.inter_dim, 3, 1)
        # 3*3卷积改成1*1卷积
        # self.conv = Conv(self.inter_dim, self.inter_dim, 1, 1)
        # self.conv = nn.Conv2d(in_channels=self.inter_dim, out_channels=512, kernel_size=1, stride=1,
        #                       bias=False)

        # # DW卷积
        # self.c1 = nn.Conv2d(in_channels=self.inter_dim, out_channels=self.inter_dim, kernel_size=1, stride=1,
        #                     groups=self.inter_dim, bias=False)
        # PW卷积
        # self.c2 = nn.Conv2d(in_channels=self.inter_dim, out_channels=self.inter_dim, kernel_size=1, bias=False)


        # self.dsconv = DepthwiseSeparableConv(in_channels=self.inter_dim, out_channels=self.inter_dim)
        self.dsconv = DepthwiseSeparableConv(in_channels=self.inter_dim, out_channels=512)
        # self.rfca = RFCAConv(inp=self.inter_dim,oup=self.inter_dim,kernel_size=1,stride=1)
        # self.rfca = RepConvBlock(in_channels=self.inter_dim,out_channels=self.inter_dim,stride=1)
        # self.dsconv = DepthwiseSeparableConv(in_channels=self.inter_dim, out_channels=self.inter_dim)
        self.level = level
        if self.level == 2:
            # self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            # self.downsample4x = Downsample(channel[0], channel[2], scale_factor=4)
            ######rga
            # self.downsample2x = Downsample(channel[1], channel[2], scale_factor=2)
            self.downsample4x = Downsample(channel[1], channel[2], scale_factor=4)
            self.downsample8x = Downsample(channel[0], channel[2], scale_factor=8)


    def forward(self, x2,x3,x4):
        input1, input2, input3 = x2, x3, x4
        if self.level == 0:
            input2 = self.upsample2x(input2)
            # print(input1.size())   (512,28,28)
            # print(input2.size())
            input3 = self.upsample4x(input3)
            # print(input3.size())
        elif self.level == 1:
            input3 = self.upsample2x1(input3)
            input1 = self.downsample2x1(input1)
        elif self.level == 2:
            # print(input1.size())   (512,28,28)
            # print(input2.size())
            input1 = self.downsample8x(input1)
            input2 = self.downsample4x(input2)
            # input2 = self.downsample2x(input2)


            # print(input1.size())
            # print(input2.size())
            # print(input3.size())
            # print(input4.size())


        level_1_weight_v = self.weight_level_1(input1)
        # print(level_1_weight_v.size())  #torch.Size([8, 8, 7, 7])
        level_2_weight_v = self.weight_level_2(input2)
        # print(level_2_weight_v.size())  #torch.Size([8, 8, 7, 7])
        level_3_weight_v = self.weight_level_3(input3)
        # print(level_3_weight_v.size())
        # print(level_4_weight_v.size())   #torch.Size([8, 8, 7, 7])

        levels_weight_v = torch.cat((level_1_weight_v, level_2_weight_v, level_3_weight_v), 1)
        # print(levels_weight_v.size())
        # print("Shape of level_1_weight_v:", level_1_weight_v.shape)
        # print("Shape of level_2_weight_v:", level_2_weight_v.shape)
        # print("Shape of level_3_weight_v:", level_3_weight_v.shape)


        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)


        fused_out_reduced = input1 * levels_weight[:, 0:1, :, :] + \
                            input2 * levels_weight[:, 1:2, :, :] + \
                            input3 * levels_weight[:, 2:, :, :]
        # print(fused_out_reduced.size())  torch.Size([8, 512, 7, 7])


        # out = self.conv(fused_out_reduced)
        # out = self.c1(fused_out_reduced)
        # out = self.c2(fused_out_reduced)
        out = self.dsconv(fused_out_reduced)
        # out = self.rfca(fused_out_reduced)
        # print(out.size())

        return out