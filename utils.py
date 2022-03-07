import torch
import torch.nn as nn


class Focal_Loss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        criterion = nn.BCELoss()
        loss = criterion(inputs, targets)
        pt = torch.exp(-loss)
        F_loss = self.alpha * (1-pt)**self.gamma * loss
        return F_loss
