import torch.nn as nn
from torch import Tensor
from mmcv.cnn import ConvModule
from mmseg.models.backbones.swin import SwinBlockSequence
from typing import List


class Representations:
    def __init__(
        self, illumination: Tensor, reflectance: Tensor, features: Tensor, clip=True
    ) -> None:
        if clip:
            self.illumination: Tensor = illumination.clip(0, 1)
            self.reflectance: Tensor = reflectance.clip(0, 1)
        else:
            self.illumination: Tensor = illumination
            self.reflectance: Tensor = reflectance
        self.features = features

    def retinex(self, ref=None, ill=None) -> Tensor:
        ref = self.reflectance if ref is None else ref
        ill = self.illumination if ill is None else ill
        return ref * ill


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        conv_cfg,
        act_cfg,
        norm_cfg,
        padding_mode,
    ):
        super().__init__()
        self.block = nn.Sequential(
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                act_cfg=act_cfg,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode,
            ),
            ConvModule(
                in_channels=channels,
                out_channels=channels,
                kernel_size=3,
                padding=1,
                conv_cfg=conv_cfg,
                act_cfg=None,
                norm_cfg=norm_cfg,
                padding_mode=padding_mode,
            ),
        )

    def forward(self, x):
        return x + self.block(x)


class LightAttention(SwinBlockSequence):
    def __init__(
        self,
        in_channels,
        out_channels,
        pre_downsample=1,
        num_heads=4,
        window_size=8,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0,
        attn_drop_rate=0,
        drop_path_rate=0,
        act_cfg=dict(type="GELU"),
        norm_cfg=dict(type="LN"),
        with_cp=False,
        init_cfg=None,
    ):
        embed_dims = out_channels * 2 * (2**pre_downsample)
        feedforward_channels = embed_dims * 4
        depth = 2
        super().__init__(
            embed_dims,
            num_heads,
            feedforward_channels,
            depth,
            window_size,
            qkv_bias,
            qk_scale,
            drop_rate,
            attn_drop_rate,
            drop_path_rate,
            None,
            act_cfg,
            norm_cfg,
            with_cp,
            init_cfg,
        )
        self.in_transform = ConvModule(
            in_channels, embed_dims, kernel_size=1, act_cfg=act_cfg
        )
        self.out_transform = nn.Sequential(
            nn.Linear(embed_dims, out_channels), nn.Sigmoid()
        )
        self.pre_downsample = pre_downsample
        if pre_downsample > 0:
            self.down = nn.MaxPool2d(kernel_size=pre_downsample, stride=pre_downsample)
            self.up = nn.UpsamplingBilinear2d(scale_factor=pre_downsample)

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_downsample > 0:
            x = self.down(x)
        x = self.in_transform(x)
        N, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(N, H * W, C)
        x, _, _, _ = super().forward(x, [H, W])
        x = self.out_transform(x)
        C = x.shape[-1]
        x = x.view(N, H, W, C).permute(0, 3, 1, 2).contiguous()
        if self.pre_downsample > 0:
            x = self.up(x)
        return x


def list_detach(feats: List[Tensor]) -> List[Tensor]:
    if isinstance(feats, list):
        return [x.detach() for x in feats]
    else:
        return feats.detach()


def losses_weight_rectify(losses: dict, prefix: str, weight: float):
    for loss_name, loss_value in losses.items():
        if loss_name.startswith(prefix):
            if "loss" in loss_name:
                losses[loss_name] = loss_value * weight
