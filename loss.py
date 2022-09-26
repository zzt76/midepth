import torch
import torch.nn as nn
import torch.nn.functional as F


class SILogLoss(nn.Module):
    def __init__(self, alpha=10.0, lamb=0.85):
        '''Scale invariant loss'''
        super(SILogLoss, self).__init__()
        self.name = "SILoss"
        self.alpha = alpha
        self.lamb = lamb

    def forward(self, pred, gt, mask, interpolate=False):
        '''Pixel value could not be 0'''
        if interpolate:
            pred = F.interpolate(pred, gt.shape[-2:], mode='bilinear', align_corners=True)
        log_diff = torch.log(pred[mask]) - torch.log(gt[mask])
        loss = torch.sqrt(torch.mean(log_diff ** 2) - self.lamb * (torch.mean(log_diff) ** 2)) * self.alpha
        return loss
