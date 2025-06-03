from mmdet.models.registry import HEADS
import torch
import torch.nn as nn
import torch.nn.functional as F
from .newcrf_layers import NewCRF
from tools.utils_newcrfs import silog_loss
from .FaPN.CAtt import CoordAtt
from .FaPN.FaPN import FeatureAlign_V2
from .FaPN.detectron_patch import Conv2d, get_norm

@HEADS.register_module
class NewcrfsDecoder(nn.Module):

    def __init__(self, dataset=None, inv_depth=False, pretrained=None, 
                    norm="BN", min_depth=0.1, max_depth=100.0, with_fapn=True, fapn_att='CA', in_channels=None, **kwargs):
        super(NewcrfsDecoder, self).__init__()
        self.inv_depth = inv_depth
        self.with_auxiliary_head = False
        self.with_neck = False
        self.dataset = dataset
        self.with_fapn = with_fapn

        if in_channels is None:
            in_channels = [24, 32, 96, 320]
        embed_dim = in_channels[3]

        win = 7

        crf_dims = [32, 64, 128, 256]
        v_dims = [16, 32, 64, embed_dim] 
        self.crf3 = NewCRF(input_dim=in_channels[3], embed_dim=crf_dims[3], window_size=win, v_dim=v_dims[3], num_heads=32)
        self.crf2 = NewCRF(input_dim=in_channels[2], embed_dim=crf_dims[2], window_size=win, v_dim=v_dims[2], num_heads=16)  
        self.crf1 = NewCRF(input_dim=in_channels[1], embed_dim=crf_dims[1], window_size=win, v_dim=v_dims[1], num_heads=8)
        self.crf0 = NewCRF(input_dim=in_channels[0], embed_dim=crf_dims[0], window_size=win, v_dim=v_dims[0], num_heads=4)

        if self.with_fapn:
            self.Att4 = CoordAtt(in_channels[3], in_channels[3], norm=None)
            self.conv3 = nn.Conv2d(in_channels[3], in_channels[2], 1, padding=0)
            self.conv2 = nn.Conv2d(in_channels[2], in_channels[1], 1, padding=0)
            self.conv1 = nn.Conv2d(in_channels[1], in_channels[0], 1, padding=0)
            
            self.FA2 = FeatureAlign_V2(in_channels[2], in_channels[2], norm=get_norm(norm, in_channels[2]), att=fapn_att)
            self.FA1 = FeatureAlign_V2(in_channels[1], in_channels[1], norm=get_norm(norm, in_channels[1]), att=fapn_att)
            self.FA0 = FeatureAlign_V2(in_channels[0], in_channels[0], norm=get_norm(norm, in_channels[0]), att=fapn_att)

        self.disp_head1 = DispHead(input_dim=crf_dims[0])

        self.up_mode = 'bilinear'
        if self.up_mode == 'mask':
            self.mask_head = nn.Sequential(
                nn.Conv2d(crf_dims[0], 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 16*9, 1, padding=0))

        self.min_depth = min_depth
        self.max_depth = max_depth


    def init_weights(self):
        pass

    def upsample_mask(self, disp, mask):
        """ Upsample disp [H/4, W/4, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = disp.shape
        mask = mask.view(N, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)

        up_disp = F.unfold(disp, kernel_size=3, padding=1)
        up_disp = up_disp.view(N, 1, 9, 1, 1, H, W)

        up_disp = torch.sum(mask * up_disp, dim=2)
        up_disp = up_disp.permute(0, 1, 4, 2, 5, 3)
        return up_disp.reshape(N, 1, 4*H, 4*W)

    def forward(self, feats):
        
        if self.with_fapn:
            up3 = self.conv3(feats[3])

            e3 = self.crf3(feats[3], self.Att4(feats[3]))
            e3 = nn.PixelShuffle(2)(e3)

            up2 = self.FA2(feats[2], up3)
            e2 = self.crf2(up2, e3)
            e2 = nn.PixelShuffle(2)(e2)

            up1 = self.FA1(feats[1], self.conv2(up2))
            e1 = self.crf1(up1, e2)
            e1 = nn.PixelShuffle(2)(e1)

            up0 = self.FA0(feats[0], self.conv1(up1))
            e0 = self.crf0(up0, e1)

        else:
            e3 = self.crf3(feats[3], feats[3])
            e3 = nn.PixelShuffle(2)(e3)
            e2 = self.crf2(feats[2], e3)
            e2 = nn.PixelShuffle(2)(e2)
            e1 = self.crf1(feats[1], e2)
            e1 = nn.PixelShuffle(2)(e1)
            e0 = self.crf0(feats[0], e1)

        if self.up_mode == 'mask':
            mask = self.mask_head(e0)
            d1 = self.disp_head1(e0, 1)
            d1 = self.upsample_mask(d1, mask)
        else:
            d1 = self.disp_head1(e0, 4)

        depth = d1 * self.max_depth
        return depth
    
    def loss(self, depth_est, depth_gt):
        silog_criterion = silog_loss(variance_focus=0.85)
        if self.dataset == 'nyu':
            mask = depth_gt > 0.1
        else:
            mask = depth_gt > 0.01
        loss = silog_criterion.forward(depth_est, depth_gt, mask.to(torch.bool))
        return dict(loss_deep=loss)

class DispHead(nn.Module):
    def __init__(self, input_dim=100):
        super(DispHead, self).__init__()
        # self.norm1 = nn.BatchNorm2d(input_dim)
        self.conv1 = nn.Conv2d(input_dim, 1, 3, padding=1)
        # self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, scale):
        # x = self.relu(self.norm1(x))
        x = self.sigmoid(self.conv1(x))
        if scale > 1:
            x = upsample(x, scale_factor=scale)
        return x

def upsample(x, scale_factor=2, mode="bilinear", align_corners=False):
    """Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=scale_factor, mode=mode, align_corners=align_corners)

