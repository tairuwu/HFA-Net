import torch
from torch import nn
from torch.nn import init

import math
# class ECA(nn.Module):
#     def __init__(self,in_channel,gamma=2,b=1):
#         super(ECA, self).__init__()
#         k=int(abs((math.log(in_channel,2)+b)/gamma))
#         kernel_size=k if k % 2 else k+1
#         padding=kernel_size//2
#         self.pool=nn.AdaptiveAvgPool2d(output_size=1)
#         self.conv=nn.Sequential(
#             nn.Conv1d(in_channels=1,out_channels=1,kernel_size=kernel_size,padding=padding,bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self,x):
#         out=self.pool(x)
#         out=out.view(x.size(0),1,x.size(1))
#         out=self.conv(out)
#         out=out.view(x.size(0),x.size(1),1,1)
#         return out*x


import torch
import torch.nn as nn
from torch.nn import functional as F


######################  ECAAttention ####     start     ###############################


class ECAAttention(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1, k_size=5):
        super(ECAAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

######################  ECAAttention ####     end     ###############################

# if __name__ == '__main__':
#     from torchsummary import summary
#     from thop import profile
#
#     model = ECAAttention(kernel_size=3)
#     summary(model, (512, 7, 7), device='cpu')
#     flops, params = profile(model, inputs=(torch.randn(1, 512, 7, 7),))
#     print(f"FLOPs: {flops}, Params: {params}")