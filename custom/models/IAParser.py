import torch
import torch.nn as nn
from mmseg.models.decode_heads import UPerHead
from mmseg.models.builder import HEADS
from mmseg.ops import resize
from mmcv.cnn import ConvModule
from mmseg.core import add_prefix
from mmseg.models.decode_heads.psp_head import PPM


@HEADS.register_module()
class IAParser(UPerHead):
    def __init__(self, illumination_channels, illumination_features_channels, **kwargs):
        """
        A segmentation head capable of comprehensively considering the influence of lighting during segmentation.
        Args:
        ill_input_channels (int): The number of channels for the input illumination features.
        ill_fea_channels (int): The number of channels for the illumination features.
        weights (dict, optional): Weights.
        """
        super().__init__(**kwargs)
        self.ill_transform = nn.Sequential(
            ConvModule(
                illumination_channels,
                illumination_features_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            ConvModule(
                illumination_features_channels,
                illumination_features_channels,
                kernel_size=3,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
        )
        self.ill_ppm = PPM(
            (1, 2, 3, 6),
            illumination_features_channels,
            illumination_features_channels // 4,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners,
        )
        self.ill_bottleneck = ConvModule(
            illumination_features_channels * 2,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        self.conv_ill_seg = nn.Conv2d(self.channels, self.num_classes, 1, 1)
        self.mask_ill_layer = nn.Sequential(
            ConvModule(
                illumination_features_channels + self.channels,
                illumination_features_channels,
                kernel_size=3,
                stride=1,
                padding=1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg,
            ),
            nn.Conv2d(
                illumination_features_channels,
                self.channels,
                kernel_size=1,
                stride=1,
            ),
            nn.Sigmoid(),
        )
        self.conv_all_seg = nn.Conv2d(self.channels, self.num_classes, 1, 1)

    def forward(self, ref_feats, ill_feats, use_for_loss=False):
        """Forward function."""
        seg_feats_ref = self._forward_feature(ref_feats)
        logits_ref = self.cls_seg(seg_feats_ref)
        with torch.no_grad():
            ill_feats = resize(
                ill_feats,
                seg_feats_ref.shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
        ill_feats = self.ill_transform(ill_feats)
        seg_feats_ill = self.ill_bottleneck(
            torch.cat([ill_feats] + self.ill_ppm(ill_feats), dim=1)
        )
        logits_ill = self.conv_ill_seg(seg_feats_ill)
        mask_ill = self.mask_ill_layer(torch.cat([ill_feats, seg_feats_ref], dim=1))
        seg_feats_whole = seg_feats_ref + seg_feats_ill * mask_ill
        logits_whole = self.conv_all_seg(seg_feats_whole)
        if not use_for_loss:
            return logits_whole
        else:
            return logits_whole, logits_ref, logits_ill

    def losses(self, seg_logits, seg_label):
        logits_whole, logits_ref, logits_ill = seg_logits
        losses = dict()
        losses.update(add_prefix(super().losses(logits_ref, seg_label), "logits_ref"))
        losses.update(
            add_prefix(super().losses(logits_whole, seg_label), "logits_whole")
        )
        losses.update(add_prefix(super().losses(logits_ill, seg_label), "logits_ill"))
        return losses
