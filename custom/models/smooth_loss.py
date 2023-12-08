import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.models import LOSSES


def gradient_loss(x: torch.Tensor, direction: str):
    kernels = {
        "x": torch.FloatTensor([[0, 0], [-1, 1]]),
        "y": torch.FloatTensor([[0, -1], [0, 1]]),
    }
    kernel = kernels[direction]
    kernel = kernel.view(1, 1, 2, 2).to(x.device)
    return torch.abs(F.conv2d(x, kernel, stride=1, padding=1))


def smooth_loss_single(I: torch.Tensor, R: torch.Tensor):
    """
    https://github.com/aasharma90/RetinexNet_PyTorch.git
    """
    Ix = gradient_loss(I, "x")
    Iy = gradient_loss(I, "y")
    Rx = F.avg_pool2d(gradient_loss(R, "x"), kernel_size=3, stride=1, padding=1)
    Ry = F.avg_pool2d(gradient_loss(R, "y"), kernel_size=3, stride=1, padding=1)
    return torch.mean(Ix * torch.exp(-10 * Rx) + Iy * torch.exp(-10 * Ry))


def smooth_loss(I: torch.Tensor, R: torch.Tensor):
    R = F.interpolate(R, size=I.shape[-2:], mode="bilinear")
    N, C, H, W = I.shape
    if C == 1:
        return smooth_loss_single(I, R.mean(dim=1, keepdim=True))
    else:
        return smooth_loss_single(I.view(N * C, 1, H, W), R.view(N * C, 1, H, W))


@LOSSES.register_module()
class SmoothLoss(nn.Module):
    def __init__(self, loss_weight=1.0) -> None:
        super().__init__()
        self.name = "SmoothLoss"
        self.loss_weight = loss_weight
    
    def forward(self, x, y):
        return smooth_loss(x, y) * self.loss_weight
