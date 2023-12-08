from mmseg.models import LOSSES
import torch
import torch.nn as nn
from mmcv.runner import force_fp32


@LOSSES.register_module()
class PixelLoss(nn.Module):
    def __init__(
        self,
        loss_weight=1.0,
        loss_type="L1",
    ):
        super().__init__()
        relation = dict(
            L1=nn.L1Loss,
            L2=nn.MSELoss,
        )
        self.name = f"PixelLoss_{loss_type}"
        self.creterion = relation[loss_type]()
        self.loss_weight = loss_weight

    @force_fp32(apply_to=("generated",))
    def __call__(
        self, generated: torch.Tensor, ground_truth: torch.Tensor, mask=None
    ) -> torch.Tensor:
        assert (
            generated.shape[-2:] == ground_truth.shape[-2:]
        ), f"{generated.shape}!={ground_truth.shape}"
        assert (
            mask is None or generated.shape[-2:] == mask.shape[-2:]
        ), f"{generated.shape}!={mask.shape}"
        if mask is None:
            return self.creterion(generated, ground_truth) * self.loss_weight
        else:
            return (
                self.creterion(generated * mask, ground_truth * mask) * self.loss_weight
            )
