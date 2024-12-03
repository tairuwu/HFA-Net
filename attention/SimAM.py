###################### PolarizedSelfAttention     ####     start    ###############################
import torch
import torch.nn as nn
from torch.nn import functional as F


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()

        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda

    def __repr__(self):
        s = self.__class__.__name__ + '('
        s += ('lambda=%f)' % self.e_lambda)
        return s

    @staticmethod
    def get_module_name():
        return "simam"

    def forward(self, x):
        b, c, h, w = x.size()

        n = w * h - 1

        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)

###################### PolarizedSelfAttention    ####     end    ###############################

###无参数注意力机制（PfAAM），在yolov5中的使用
# class PfAAMLayer(nn.Module):
#     def __init__(self, c1, c2, ratio=16):
#         super(PfAAMLayer, self).__init__()
#         self.avg_pool = nn.AdaptiveAvgPool2d(1)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.avg_pool(x).view(b, c, 1, 1).expand_as(x)
#         z = torch.mean(x, dim=1, keepdim=True).expand_as(x)
#         return x * self.sigmoid(y * z)





class PfAAMLayer(nn.Module):
    def __init__(self):
        super(PfAAMLayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        # z = torch.mean(x, 1)
        # return x * self.sigmoid(y * z)
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c, 1, 1).expand_as(x)
        z = torch.mean(x, dim=1, keepdim=True).expand_as(x)
        return x * self.sigmoid(y * z)

