from model.RMT import RMT_modify

from model.mul_block import MSA
from module import MLFA
import torch.nn as nn
import torch
from timm.models.layers import trunc_normal_
from module import AAFF



class DualInception(nn.Module):
    def __init__(self, num_classes = 5):
        super(DualInception, self).__init__()
        self.msa = MSA()
        self.rmt = RMT_modify()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.msff = MLFA(inter_dim=1024, level=2, channel=[128, 256, 512,1024])
        self.aff = AAFF(1024, 512)

        self.head = nn.Linear(512, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        # 定义了一个辅助方法 _init_weights，用于初始化模型的权重。
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            try:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
            except:
                pass

    def forward(self, x):

        m1, m2, m3, m4 = self.msa(x)
        t1, t2, t3, t4 = self.rmt(x)

        mt1 = torch.cat((m1, t1), dim=1)
        mt2 = torch.cat((m2 ,t2) ,dim=1)
        mt3 = torch.cat((m3, t3), dim=1)
        mt4 = torch.cat((m4, t4), dim=1)

        mt = self.msff(mt1, mt2, mt3, mt4)


        output = self.aff(mt4, mt)

        output = self.avgpool(output)  # B C 1
        output = torch.flatten(output, 1)
        output = self.head(output)

        return output
