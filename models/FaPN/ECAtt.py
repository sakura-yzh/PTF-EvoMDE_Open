from math import log
import torch
from torch import nn
from .detectron_patch import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

class eca_layer(nn.Module):
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, oup, norm,  k_size=3, gamma=2, b=1):
        super(eca_layer, self).__init__()
        t= int(abs((log(channel,2)+b)/gamma))
        k = t if t % 2 else t+1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False) 
        self.sigmoid = nn.Sigmoid()

        self.conv1 = Conv2d(channel, oup, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.conv1)

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)
        out = x * y.expand_as(x)

        return self.conv1(out)