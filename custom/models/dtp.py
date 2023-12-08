import torch
from mmseg.models.segmentors import EncoderDecoder
from mmseg.ops import resize
from mmseg.models.builder import SEGMENTORS, build_head, build_loss
from .utils import Representations
from .SOD_head import SODHead
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from mmcv.runner import auto_fp16
from mmseg.core import add_prefix
from functools import partial
from torch import Tensor


@SEGMENTORS.register_module()
class DTP(EncoderDecoder):
    def __init__(
        self,
        backbone,
        decode_head,
        disentangle_head,
        dark_class=[10],
        disturb_beta="uniform",  # for debug
        disentangle_loss=None,
        disturb_mode=["direct", "random", "direct", "max", "direct"],
        eps=1e-6,
        auxiliary_head=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        init_cfg=None,
    ):
        super().__init__(
            backbone, decode_head, None, None, train_cfg, test_cfg, pretrained, init_cfg
        )
        self.disentangle_head: SODHead = build_head(disentangle_head)
        self.eps = eps
        self.dark_class = dark_class
        self.disturb_mode = disturb_mode
        self.idx = 0
        self.disentanlge_loss = build_loss(disentangle_loss)

    def norm_imnet2std(self, img: Tensor) -> Tensor:
        return (
            img * img.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)
            + img.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        ).clamp_min(0) / 255.0

    def norm_std2imnet(self, img: Tensor) -> Tensor:
        return (
            img * 255.0 - img.new_tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
        ) / img.new_tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)

    # Illumination estimation for certain classes may be unneeded
    @torch.no_grad()
    def get_retinex_mask(self, gt_semantic_seg: Tensor, mask_shape=None) -> Tensor:
        mask = torch.zeros_like(gt_semantic_seg, dtype=torch.bool)
        for i in self.dark_class:
            mask = mask | gt_semantic_seg.eq(i)
        mask = ~mask
        if mask_shape is not None:
            mask = resize(
                mask.to(dtype=torch.float),
                size=mask_shape,
                mode="nearest",
            ).to(dtype=torch.bool)
        return mask

    def get_disturb_mode(self):
        return np.random.choice(self.disturb_mode)

    def disturb(self, illumination: Tensor, noise: Tensor) -> Tensor:
        self.idx += 1
        beta = max(1 - self.idx / 80000, 0)
        beta = np.random.uniform(beta, 1)
        return (1 - beta) * illumination + beta * noise

    def train_step(self, data_batch, optimizer, **kwargs):
        outputs = dict(loss=[], log_vars=dict())
        losses = self.forward_train(**data_batch)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data_batch["metasA"])
        )

        return outputs

    def generate_random_illumination(self, reference: Tensor) -> Tensor:
        x = torch.randn_like(reference)
        N, C, H, W = x.shape
        k = 32
        x = torch.einsum("nchpwq->nchwpq", x.reshape(N, C, H // k, k, W // k, k))
        x = x.mean(dim=[4, 5])
        x = (x - x.min()) / (x.max() - x.min())
        x.clip_(0.1, 1)
        return x

    def generate_max_illumination(self, im: Tensor) -> Tensor:
        x = im.max(dim=1, keepdim=True).values
        x = F.interpolate(x, scale_factor=1 / 16, mode="bilinear")
        return F.interpolate(x, scale_factor=16, mode="bilinear")

    def IDWE(
        self,
        imA: Tensor,
        imB: Tensor,
        repreA: Representations,
        repreB: Representations,
        mode: str,
    ) -> Tuple[Tensor, Tensor]:
        with torch.no_grad():
            if mode == "direct":
                noise = repreB.illumination
            elif mode == "random":
                noise = self.generate_random_illumination(repreA.illumination)
            else:
                noise = self.generate_max_illumination(imB)
        noise = resize(
            noise,
            repreA.illumination.shape[-2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        ill = self.disturb(repreB.illumination, noise)
        disturbed_img = repreA.retinex(ill=ill)
        return disturbed_img, ill.clone().detach()

    def encode_decode(self, img: Tensor, img_metas):
        img = self.norm_imnet2std(img)
        repres = self.disentangle_head.forward(img)
        feats = self.extract_feat(self.norm_std2imnet(repres.reflectance))
        seg = self.decode_head(feats, repres.features)
        out = resize(
            input=seg,
            size=img.shape[2:],
            mode="bilinear",
            align_corners=self.align_corners,
        )
        return out

    @auto_fp16(
        apply_to=(
            "imgA",
            "imgB",
        )
    )
    def forward_train(
        self,
        imgA: Tensor,
        gtA: Tensor,
        metasA,
        imgB: Tensor,
        gtB: Tensor,
        metasB,
    ):
        self.metas = metasA
        mode = self.get_disturb_mode()
        losses = dict()
        imgA, imgB = map(self.norm_imnet2std, [imgA, imgB])
        repreA, disentanlge_lossA = self.disentangle_head.forward_train(imgA)
        repreB, disentanlge_lossB = self.disentangle_head.forward_train(imgB)
        im_RAIB, I_RAIB = self.IDWE(imgA, imgB, repreA, repreB, mode)
        im_RBIA, I_RBIA = self.IDWE(imgB, imgA, repreB, repreA, mode)
        repre_RAIB, disentanlge_lossRAIB = self.disentangle_head.forward_train(im_RAIB)
        repre_RBIA, disentanlge_lossRBIA = self.disentangle_head.forward_train(im_RBIA)
        featsA, featsB, featsRAIB, featsRBIA = map(
            self.extract_feat,
            map(
                self.norm_std2imnet,
                [
                    repreA.reflectance,
                    repreB.reflectance,
                    repre_RAIB.reflectance,
                    repre_RBIA.reflectance,
                ],
            ),
        )
        segA, segB, segRAIB, segRBIA = map(
            partial(self.decode_head.forward, use_for_loss=True),
            [featsA, featsB, featsRAIB, featsRBIA],
            [
                repreA.features.detach(),
                repreB.features.detach(),
                repre_RAIB.features.detach(),
                repre_RBIA.features.detach(),
            ],
        )
        losses.update(add_prefix(self.decode_head.losses(segA, gtA), "segA"))
        losses.update(add_prefix(self.decode_head.losses(segB, gtB), "segB"))
        losses.update(add_prefix(self.decode_head.losses(segRAIB, gtA), "segRAIB"))
        losses.update(add_prefix(self.decode_head.losses(segRBIA, gtB), "segRBIA"))
        losses.update(add_prefix(disentanlge_lossA, "SOD.A"))
        losses.update(add_prefix(disentanlge_lossB, "SOD.B"))
        losses.update(add_prefix(disentanlge_lossRAIB, "SOD.RAIB"))
        losses.update(add_prefix(disentanlge_lossRBIA, "SOD.RBIA"))
        # Given that we've established a constraint where the multiplication of illumination and reflectance equals the original image, imposing consistency constraints on illumination makes the constraint on reflectance actually unnecessary. Consequently, this step has been excluded in this reproduced version.
        losses["loss_disentangleA"] = self.disentanlge_loss(
            repre_RBIA.illumination,
            I_RBIA,
            self.get_retinex_mask(gtA, repreA.illumination.shape[-2:]),
        )
        losses["loss_disentangleB"] = self.disentanlge_loss(
            repre_RAIB.illumination,
            I_RAIB,
            self.get_retinex_mask(gtB, repreB.illumination.shape[-2:]),
        )
        return losses
