import torch
import torch.nn as nn
from torch import Tensor
from mmseg.models.builder import LOSSES
from mmcv.cnn import ConvModule
from mmcv.runner import BaseModule, auto_fp16
from .utils import Representations, ResidualBlock, LightAttention
from mmseg.ops import resize
from typing import Tuple
from mmseg.models.builder import HEADS


@HEADS.register_module()
class SODHead(BaseModule):
    def __init__(
        self,
        channels,
        in_channels=3,
        base_channels=32,
        num_downsample=2,
        num_resblock=2,
        attn_channels=16,
        image_pool_channels=32,
        ill_embeds_op="+",
        clip=True,
        gray_illumination=False,
        eps=1e-5,
        loss_retinex=None,
        loss_smooth=None,
        conv_cfg=None,
        norm_cfg=dict(type="IN2d"),
        act_cfg=dict(type="ReLU"),
        align_corners=False,
        init_cfg=dict(type="Normal", std=0.01),
    ):
        super().__init__(init_cfg)
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.align_corners = align_corners
        self.fp16_enabled = False
        self.loss_retinex = LOSSES.build(loss_retinex)
        self.loss_smooth = LOSSES.build(loss_smooth)
        self.eps = eps
        self.init_autoencoder(
            base_channels,
            num_downsample,
            num_resblock,
            attn_channels,
            image_pool_channels,
        )
        self.reflectance_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Tanh(),
        )
        self.illumination_output = nn.Sequential(
            nn.Conv2d(
                self.channels,
                3,
                kernel_size=7,
                padding=3,
                padding_mode="reflect",
            ),
            nn.Sigmoid(),
        )
        self.ill_embeds_op = ill_embeds_op
        self.clip = clip
        self.gray_illumination = gray_illumination

    def init_autoencoder(
        self,
        base_channels,
        num_downsample,
        num_resblock,
        attn_channels,
        image_pool_channels,
    ):
        assert (
            num_resblock >= 1
            and num_downsample >= 1
            and attn_channels >= 1
            and image_pool_channels >= 1
        )
        channels = base_channels
        self.stem = ConvModule(
            self.in_channels,
            channels,
            kernel_size=7,
            padding=3,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        down_layers = []
        for _ in range(num_downsample):
            down_layers += [
                ConvModule(
                    channels,
                    channels * 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                )
            ]
            channels *= 2
        self.downsample = nn.Sequential(*down_layers)
        res_layers = []
        for _ in range(num_resblock):
            res_layers += [
                ResidualBlock(
                    channels,
                    self.conv_cfg,
                    self.act_cfg,
                    self.norm_cfg,
                    "reflect",
                )
            ]
        self.residual = nn.Sequential(*res_layers)
        self.light_attention = LightAttention(channels, attn_channels, pre_downsample=2)
        self.image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, image_pool_channels, kernel_size=1),
            nn.ReLU(inplace=True),
        )
        self.merge = ConvModule(
            channels + attn_channels + image_pool_channels,
            channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )
        upsample_ill_layers, upsample_ref_layers = [], []
        for _ in range(num_downsample):
            upsample_ill_layers += [
                ConvModule(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
            ]
            upsample_ref_layers += [
                nn.Upsample(
                    scale_factor=2, mode="bilinear", align_corners=self.align_corners
                ),
                ConvModule(
                    channels,
                    channels // 2,
                    kernel_size=3,
                    padding=1,
                    padding_mode="reflect",
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                ),
            ]
            channels //= 2
        upsample_ref_layers += [nn.Conv2d(channels, channels, kernel_size=1)]
        self.upsample_illumination = nn.Sequential(*upsample_ill_layers)
        self.upsample_reflectance = nn.Sequential(*upsample_ref_layers)
        self.refine_reflectance = ConvModule(
            base_channels,
            base_channels,
            kernel_size=3,
            padding=1,
            padding_mode="reflect",
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
        )

    def _forward_feature(self, imgs: Tensor) -> Tensor:
        img_embeds = self.stem(imgs)
        feats = [self.downsample(img_embeds)]
        feats += [
            resize(
                self.image_pool(feats[0]),
                size=feats[0].shape[-2:],
                mode="bilinear",
                align_corners=self.align_corners,
            ),
            self.light_attention(feats[0]),
        ]
        feats = torch.cat(feats, dim=1)
        feats = self.merge(feats)
        feats = self.residual(feats)
        ill_embeds = self.upsample_illumination(feats)
        ref_embeds = self.upsample_reflectance(feats)
        if self.ill_embeds_op == "+":
            ref_embeds = self.refine_reflectance(ref_embeds + ill_embeds + img_embeds)
        elif self.ill_embeds_op == "-":
            ref_embeds = self.refine_reflectance(ref_embeds - ill_embeds + img_embeds)
        return ref_embeds, ill_embeds, feats

    @auto_fp16(apply_to=("imgs",))
    def forward(self, imgs: Tensor) -> Representations:
        ref_embeds, ill_embeds, feats = self._forward_feature(
            torch.cat([imgs, torch.max(imgs, dim=1, keepdim=True).values], dim=1)
        )
        illumination = self.illumination_output(ill_embeds)
        illumination = torch.mean(illumination, dim=1, keepdim=True).repeat(1, 3, 1, 1)
        reflectance = self.reflectance_output(ref_embeds) + imgs
        return Representations(illumination, reflectance, feats, clip=self.clip)

    def forward_train(self, imgs: Tensor) -> Tuple[Representations, dict]:
        repres = self.forward(imgs)
        losses = dict(
            loss_smooth=self.loss_smooth(repres.illumination, repres.reflectance),
            loss_retinex=self.loss_retinex(repres.retinex(), imgs),
        )
        return repres, losses
