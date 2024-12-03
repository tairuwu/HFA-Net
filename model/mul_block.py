import torch

import torch.nn as nn
from attention.EMA import EMA

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class MulScaleBlock(nn.Module):
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(MulScaleBlock, self).__init__()
        norm_layer = nn.BatchNorm2d
        scale_width = int(planes / 4)

        self.scale_width = scale_width

        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=False)

        self.conv1_2_1 = conv3x3(scale_width, scale_width)
        self.bn1_2_1 = norm_layer(scale_width)
        self.conv1_2_2 = conv3x3(scale_width, scale_width)
        self.bn1_2_2 = norm_layer(scale_width)
        self.conv1_2_3 = conv3x3(scale_width, scale_width)
        self.bn1_2_3 = norm_layer(scale_width)
        self.conv1_2_4 = conv3x3(scale_width, scale_width)
        self.bn1_2_4 = norm_layer(scale_width)

        self.att = EMA(channels=planes, factor=32)


        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        sp_x = torch.split(out, self.scale_width, 1)

        out_1_1 = self.bn1_2_1(self.conv1_2_1(sp_x[0]))
        out_1_2 = self.bn1_2_2(self.conv1_2_2(self.relu(out_1_1) + sp_x[1]))
        out_1_3 = self.bn1_2_3(self.conv1_2_3(self.relu(out_1_2) + sp_x[2]))
        out_1_4 = self.bn1_2_4(self.conv1_2_4(self.relu(out_1_3) + sp_x[3]))
        out_1 = torch.cat([out_1_1, out_1_2, out_1_3, out_1_4], dim=1)
        out =self.att(out_1)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class MSA(nn.Module):

    def __init__(self, block_b=MulScaleBlock, layers=[1, 1, 1, 1], num_classes=1000, zero_init_residual=False):
        super(MSA, self).__init__()
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = norm_layer(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block_b, 64, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block_b, 64, 128, layers[1], stride=2)
        # In this branch, each BasicBlock replaced by MulScaleBlock.
        self.layer3 = self._make_layer(block_b, 128, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block_b, 256, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))


        self.fc_1 = nn.Linear(512, num_classes)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, block_b):
                    nn.init.constant_(m.bn2.weight, 0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(conv1x1(inplanes, planes, stride), norm_layer(planes))
        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes))
        return nn.Sequential(*layers)

    def _forward_impl(self, x):

        of_part = x


        of_part = self.conv1(of_part)
        of_part = self.bn1(of_part)
        of_part = self.relu(of_part)
        of_part = self.maxpool(of_part)
        of_part = self.layer1(of_part)
        x1 = of_part
        # print(x1.size())

        of_part = self.layer2(of_part)
        x2 = of_part

        # branch 2 ############################################
        of_part = self.layer3(of_part)
        x3 = of_part

        of_part = self.layer4(of_part)  #(8,512,14,14)
        x4 = of_part

        return x1,x2,x3,x4
        # return output

    def forward(self, x):
        return self._forward_impl(x)


