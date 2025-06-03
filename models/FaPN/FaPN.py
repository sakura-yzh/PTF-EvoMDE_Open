import torch.nn.functional as F
import torch
from torch import nn
from dcn_v2 import DCN as dcn_v2
import fvcore.nn.weight_init as weight_init
from .detectron_patch import Conv2d, get_norm
from .CAtt import CoordAtt
from .se_block import SEBlock
from .ECAtt import eca_layer

class FeatureSelectionModule(nn.Module):
    def __init__(self, in_chan, out_chan, norm="GN"):
        super(FeatureSelectionModule, self).__init__()
        self.conv_atten = Conv2d(in_chan, in_chan, kernel_size=1, bias=False, norm=get_norm(norm, in_chan))
        self.sigmoid = nn.Sigmoid()
        self.conv = Conv2d(in_chan, out_chan, kernel_size=1, bias=False, norm=get_norm('', out_chan))
        weight_init.c2_xavier_fill(self.conv_atten)
        weight_init.c2_xavier_fill(self.conv)

    def forward(self, x):
        atten = self.sigmoid(self.conv_atten(F.avg_pool2d(x, x.size()[2:])))
        feat = torch.mul(x, atten)
        x = x + feat
        feat = self.conv(x)
        return feat


class FeatureAlign_V2(nn.Module):  # FaPN full version
    def __init__(self, in_nc=128, out_nc=128, norm=None, att='CA'):
        super(FeatureAlign_V2, self).__init__()
        if att == 'FSM':
            self.lateral_conv = FeatureSelectionModule(in_nc, out_nc, norm="")  # FSM
        elif att == 'CA':
            self.lateral_conv = CoordAtt(in_nc, out_nc, norm="")  # CA
        elif att == 'SE':
            self.lateral_conv = SEBlock(in_nc, out_nc, norm="")  # SE
        # elif att == 'ECA':
        #     self.lateral_conv = eca_layer(in_nc, out_nc, norm="")  # ECA
        else:
            assert False,'fapn attention is error!'
        self.offset = Conv2d(out_nc * 2, out_nc, kernel_size=1, stride=1, padding=0, bias=False, norm=norm)
        self.dcpack_L2 = dcn_v2(out_nc, out_nc, 3, stride=1, padding=1, dilation=1, deformable_groups=8,
                                extra_offset_mask=True)
        self.relu = nn.ReLU(inplace=True)
        weight_init.c2_xavier_fill(self.offset)

    def forward(self, feat_l, feat_s, main_path=None):
        HW = feat_l.size()[2:]
        if feat_l.size()[2:] != feat_s.size()[2:]:
            feat_up = F.interpolate(feat_s, HW, mode='bilinear', align_corners=False)
        else:
            feat_up = feat_s
        feat_arm = self.lateral_conv(feat_l)  # 0~1 * feats
        # print("feat_arm", feat_arm.shape)
        # print("feat_up", feat_up.shape)
        offset = self.offset(torch.cat([feat_arm, feat_up * 2], dim=1))  # concat for offset by compute the dif
        feat_align = self.relu(self.dcpack_L2([feat_up, offset], main_path))  # [feat, offset]
        return feat_align + feat_arm
    
