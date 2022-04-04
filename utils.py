import os
import sys
import torch
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from torch.nn.utils.rnn import pad_sequence
from sklearn import metrics
from transformers import BertTokenizer


class Focal_Loss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super(Focal_Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        # criterion = nn.BCELoss()
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        alphas = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alphas * (1-pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)


class Focal_Loss_Teacher(nn.Module):
    def __init__(self, index, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.index = index

    def forward(self, inputs, targets):
        # criterion = nn.BCELoss()
        teacher_task_mask = torch.zeros_like(inputs)
        teacher_task_mask[:, self.index] = 1
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        alphas = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alphas * (1-pt)**self.gamma * BCE_loss * teacher_task_mask
        return torch.mean(F_loss)


class Focal_Loss_Teacher(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.criterion = Focal_Loss()

    def forward(self, inputs, targets, distill_weight=0.5):
        targets_true, targets_teacher = torch.split(targets, targets.shape[-1]//2, dim=-1)
        loss_true = self.criterion(inputs, targets_true)
        loss_distill = self.criterion(inputs, targets_teacher)
        return (1 - distill_weight) * loss_true + distill_weight * loss_distill


def load_tokenizer(bert_model_name):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"
    return tokenizer


def auc(y_pred, y_true):
    return metrics.roc_auc_score(y_true, y_pred)


def local_evaluate(submission_df, test_labels, columns):
    if isinstance(test_labels, str):
        test_labels = pd.read_csv(test_labels)
    test_labels = test_labels.merge(submission_df, how='left', on='id', suffixes=['_true', '_pred'])
    columns_auc = []
    for col in columns:
        columns_auc.append(auc(test_labels[col + '_pred'], test_labels[col + '_true']))
    columns_auc.append(sum(columns_auc) / len(columns_auc))
    return columns_auc


def generate_teacher_labels(model, column_name, column_suffix,
                            tokenizer, state_dict_path,
                            csv_path, device, batch_size=32):
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    index = np.where(np.array(columns) == column_name)[0][0]
    model.load_state_dict(torch.load(state_dict_path, map_location=device))
    model.eval()

    test_df = pd.read_csv(csv_path)
    teacher_column = np.zeros((len(test_df)))
    for i in range(len(test_df) // batch_size + 1):
        batch_df = test_df.iloc[i * batch_size: (i + 1) * batch_size]
        texts = []
        for text in batch_df["comment_text"].tolist():
            text = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
            texts.append(torch.LongTensor(text))
        x = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=mask)
        outputs = outputs.cpu().numpy()
        teacher_column[i * batch_size: (i + 1) * batch_size] = outputs[:, index]
    test_df[column_name + column_suffix] = teacher_column
    return test_df


if __name__ == '__main__':
    # generate teacher labels
    parser = argparse.ArgumentParser(description='Toxic Comments Classification')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--state_dict_path', type=str)
    parser.add_argument('--bertname', type=str, default='bert-base-uncased')
    parser.add_argument('--column_name', type=str,
                        choices=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--num_classes', type=int, default=6)
    args = parser.parse_args()

    tokenizer = load_tokenizer('bert-base-uncased')
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    from model import BertClassifier
    model = BertClassifier(args).to(device)

    res_df = generate_teacher_labels(model, args.column_name, args.suffix,
                                     tokenizer, args.state_dict_path,
                                     csv_path='./train.csv', device=device)
    res_df.to_csv('./teachers_' + args.column_name + '.csv', index=False)
