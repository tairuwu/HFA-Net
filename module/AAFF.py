
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn import functional as F
class AAFF(nn.Module):

    def __init__(self, input_channels, internal_neurons):
        super(AAFF, self).__init__()
        self.fc1 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.fc2 = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1, bias=True)
        self.cov = nn.Conv2d(in_channels=input_channels, out_channels=input_channels, kernel_size=1, stride=1,
                             bias=False)
        # self.dropout = nn.Dropout(p=0.1)  # 添加一个dropout层，p表示丢弃概率
        self.cov2 = nn.Conv2d(in_channels=input_channels, out_channels=internal_neurons, kernel_size=1, stride=1, bias=False)
        # self.input_channels = input_channels

    def forward(self, inputs,mt):
        inputs = self.cov2(inputs)
        inputs = torch.cat((inputs, mt), dim=1)
        x1 = F.adaptive_avg_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x1 = self.fc1(x1)
        x1 = F.relu(x1, inplace=True)
        # x1 = self.dropout(x1)  # 添加dropout层
        x1 = self.fc2(x1)
        x1 = torch.sigmoid(x1)


        x2 = F.adaptive_max_pool2d(inputs, output_size=(1, 1))
        # print('x:', x.shape)
        x2 = self.fc1(x2)
        x2 = F.relu(x2, inplace=True)
        # x2 = self.dropout(x2)  # 添加dropout层
        x2 = self.fc2(x2)
        x2 = torch.sigmoid(x2)
        x = x1*inputs + x2*inputs
        # x = x + mt
        x = self.cov2(x)
        return x
