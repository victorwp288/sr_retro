import torch
from torch import nn
import torch.nn.functional as F


class SobelLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32)
        self.register_buffer("kernel_x", kernel_x.view(1, 1, 3, 3))
        self.register_buffer("kernel_y", kernel_y.view(1, 1, 3, 3))
        self.epsilon = epsilon

    def forward(self, prediction, target):
        channels = prediction.size(1)
        kernel_x = self.kernel_x.to(prediction).repeat(channels, 1, 1, 1)
        kernel_y = self.kernel_y.to(prediction).repeat(channels, 1, 1, 1)
        grad_pred_x = F.conv2d(prediction, kernel_x, padding=1, groups=channels)
        grad_pred_y = F.conv2d(prediction, kernel_y, padding=1, groups=channels)
        grad_tgt_x = F.conv2d(target, kernel_x, padding=1, groups=channels)
        grad_tgt_y = F.conv2d(target, kernel_y, padding=1, groups=channels)
        mag_pred = torch.sqrt(grad_pred_x * grad_pred_x + grad_pred_y * grad_pred_y + self.epsilon)
        mag_tgt = torch.sqrt(grad_tgt_x * grad_tgt_x + grad_tgt_y * grad_tgt_y + self.epsilon)
        return torch.mean(torch.abs(mag_pred - mag_tgt))
