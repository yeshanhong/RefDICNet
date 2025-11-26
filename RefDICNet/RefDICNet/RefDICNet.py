import torch
import torch.nn.functional as F
import torch.nn as nn

from RefDICNet import transformer

from RefDICNet import corr
from RefDICNet import refine
from RefDICNet import upsample
from RefDICNet import config
from RefDICNet.backbone import ResNetEncoder
from RefDICNet.refine import BasicUpdateBlock

class MergeS8(nn.Module):
    def __init__(self, dim_s4, dim_s2, x0_dim):
        super().__init__()
        in_channels = dim_s4 + dim_s2 + x0_dim

        self.edge_enhancer = nn.Sequential(
            nn.Conv2d(x0_dim, dim_s2, 1, bias=False),
            nn.BatchNorm2d(dim_s2),
            nn.GELU()
        )
        # self.residual = nn.Conv2d(in_channels, dim_s2, 1, bias=False)
        self.final_conv = nn.Sequential(
            nn.Conv2d(in_channels, dim_s2, 3, padding=1, bias=False),
            # nn.BatchNorm2d(dim_s2),
            nn.GELU(),
            nn.Conv2d(dim_s2, dim_s2, 3, padding=1, bias=False),
            nn.BatchNorm2d(dim_s2)
        )

    def forward(self, features_s2, features_s4, x0):
        x = torch.cat([features_s2, features_s4, x0], dim=1)
        edge_feat = self.edge_enhancer(x0)
        x = self.final_conv(x)
        return x + edge_feat

class RefDICNet(torch.nn.Module):
    def __init__(self):
        super(RefDICNet, self).__init__()

        self.backbone = ResNetEncoder(output_dim=192, num_output_scales=2)
        self.Transformer = transformer.FeatureAttention(config.feature_dim_s4 + config.context_dim_s4,
                                                           num_layers=2, ffn=True, ffn_dim_expansion=1, post_norm=True)

        self.corr_block_s4 = corr.CorrBlock(radius=4, levels=1)
        self.corr_block_s2 = corr.CorrBlock(radius=4, levels=1)

        self.merge_s2 = MergeS8(config.feature_dim_s4, config.feature_dim_s2, 128)
        self.context_merge_s2 = torch.nn.Sequential(
            torch.nn.Conv2d(config.context_dim_s4 + config.context_dim_s2, config.context_dim_s2, kernel_size=3,
                            stride=1, padding=1, bias=False),
            torch.nn.GELU(),
            torch.nn.Conv2d(config.context_dim_s2, config.context_dim_s2, kernel_size=3, stride=1, padding=1,
                            bias=False),
            torch.nn.BatchNorm2d(config.context_dim_s2))

        self.refine_s4 = BasicUpdateBlock(corr_levels=1, corr_radius=4, hidden_dim=128)
        self.refine_s2 = BasicUpdateBlock(corr_levels=1, corr_radius=4, hidden_dim=128)

        self.mask = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2 ** 2 * 9, 1, padding=0))

        self.upsample_s2 = upsample.UpSample(64, upsample_factor=2)

        for p in self.parameters():
            if p.dim() > 1:
                torch.nn.init.xavier_uniform_(p)

    def init_bhwd(self, batch_size, height, width, device, amp=False):

        self.corr_block_s4.init_bhwd(batch_size, height // 4, width // 4, device, amp)
        self.corr_block_s2.init_bhwd(batch_size, height // 2, width // 2, device, amp)

        self.init_iter_context_s4 = torch.zeros(batch_size, config.iter_context_dim_s4, height // 4, width // 4,
                                                 device=device, dtype=torch.half if amp else torch.float)
        self.init_iter_context_s2 = torch.zeros(batch_size, config.iter_context_dim_s2, height // 2, width // 2,
                                                device=device, dtype=torch.half if amp else torch.float)
        self.init_flow0 = torch.zeros(batch_size, 2, height // 4, width // 4,
                                                device=device, dtype=torch.half if amp else torch.float)

    def split_features(self, features, context_dim, feature_dim):

        context, features = torch.split(features, [context_dim, feature_dim], dim=1)

        context, _ = context.chunk(chunks=2, dim=0)
        feature0, feature1 = features.chunk(chunks=2, dim=0)

        return features, torch.relu(context)

    def forward(self, img0, img1, iters_s4=4, iters_s2=8, corr_radius=4, global_corr=True):

        flow_list = []

        img0 /= 255.
        img1 /= 255.

        features_s4, features_s2, x0 = self.backbone(torch.cat([img0, img1], dim=0))

        features_s4 = self.Transformer(features_s4)

        features_s4, context_s4 = self.split_features(features_s4, config.context_dim_s4, config.feature_dim_s4)
        features_s2, context_s2 = self.split_features(features_s2, config.context_dim_s2, config.feature_dim_s2)

        feature0_s4, feature1_s4 = features_s4.chunk(chunks=2, dim=0)
        #
        flow0 = self.init_flow0

        corr_pyr_s4 = self.corr_block_s4.init_corr_pyr(feature0_s4, feature1_s4)

        iter_context_s4 = self.init_iter_context_s4

        for i in range(iters_s4):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s4 = iter_context_s4.detach()

            corrs = self.corr_block_s4(corr_pyr_s4, flow0)

            iter_context_s4, delta_flow = self.refine_s4(corrs, context_s4, iter_context_s4, flow0)

            flow0 = flow0 + delta_flow

            if self.training:
                up_flow0 = F.interpolate(flow0, scale_factor=4, mode='bilinear') * 4
                flow_list.append(up_flow0)

        flow0 = F.interpolate(flow0, scale_factor=2, mode='nearest') * 2

        features_s4 = F.interpolate(features_s4, scale_factor=2, mode='nearest')

        features_s2 = self.merge_s2(features_s2, features_s4, x0)

        feature0_s2, feature1_s2 = features_s2.chunk(chunks=2, dim=0)

        corr_pyr_s2 = self.corr_block_s2.init_corr_pyr(feature0_s2, feature1_s2)

        context_s4 = F.interpolate(context_s4, scale_factor=2, mode='nearest')

        context_s2 = self.context_merge_s2(torch.cat([context_s2, context_s4], dim=1))

        iter_context_s2 = self.init_iter_context_s2

        for i in range(iters_s2):

            if self.training and i > 0:
                flow0 = flow0.detach()
                # iter_context_s2 = iter_context_s2.detach()

            corrs = self.corr_block_s2(corr_pyr_s2, flow0)

            iter_context_s2, delta_flow = self.refine_s2(corrs, context_s2, iter_context_s2, flow0)
            mask_up = .25 * self.mask(iter_context_s2)
            flow0 = flow0 + delta_flow

            if self.training or i == iters_s2 - 1:

                up_flow0 = self.upsample_s2(mask_up, flow0) * 2
                flow_list.append(up_flow0)

        return flow_list