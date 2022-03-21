import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from sklearn import metrics


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


def auc(y_pred, y_true):
    return metrics.roc_auc_score(y_true, y_pred)


def local_evaluate(submission_df, test_labels, columns):
    if isinstance(test_labels, str):
        test_labels = pd.read_csv(test_labels)
    test_labels = test_labels.merge(submission_df, how='left', on='id', suffixes=['_true', '_pred'])
    columns_auc = []
    for col in columns:
        columns_auc.append(auc(test_labels[col + '_pred'], test_labels[col + '_true']))
    columns_auc.append(torch.mean(columns_auc))
    return columns_auc
