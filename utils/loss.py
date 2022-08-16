import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=0, size_average=True, ignore_index=255):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.size_average = size_average

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        if self.size_average:
            return focal_loss.mean()
        else:
            return focal_loss.sum()


class FSAuxCELoss(nn.Module):
    def __init__(self, ignore_index):
        super(FSAuxCELoss, self).__init__()
        self.ignore_index = ignore_index

    def forward(self, inputs, targets):
        seg_out,aux_out  = inputs
        seg_loss = F.cross_entropy(
            seg_out, targets, reduction='mean', ignore_index=self.ignore_index)
        aux_loss = F.cross_entropy(
            aux_out, targets, reduction='mean', ignore_index=self.ignore_index)
        loss = seg_loss+0.4*aux_loss
        return loss
