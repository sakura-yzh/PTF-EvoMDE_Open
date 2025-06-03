import torch
import torch.nn as nn
from .detectron_patch import Conv2d, get_norm
import fvcore.nn.weight_init as weight_init

class SEBlock(nn.Module):
    def __init__(self, n_channels, oup, norm, rate=4):
        super(SEBlock, self).__init__()
        self.se_block = nn.Sequential(nn.Linear(n_channels, int(n_channels / rate)),
                                      nn.ReLU(),
                                      nn.Linear(int(n_channels / rate), n_channels),
                                      nn.Sigmoid())
        self.conv = Conv2d(n_channels, oup, kernel_size=1, stride=1, padding=0)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        x_gp = torch.mean(torch.mean(x, dim=-1), dim=-1)
        att = self.se_block(x_gp).unsqueeze(dim=-1).unsqueeze(dim=-1)
        return self.conv(x * att)
