import torch
import torch.nn as nn
import torch.nn.functional as F


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        #criterion = nn.BCELoss()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        alphas = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alphas * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)
