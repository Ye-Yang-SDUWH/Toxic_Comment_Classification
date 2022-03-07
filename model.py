import torch
import os
import torch.nn as nn
from utils import Focal_Loss
from capsnet import Capsule
from transformers import BertTokenizer, BertModel, AutoConfig


def load_tokenizer(bert_model_name):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"
    return tokenizer


class BertClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert = BertModel.from_pretrained(args.bertname)
        self.classifier = nn.Linear(self.bert.config.hidden_size, args.num_classes)


    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = outputs[1]  # batch, hidden
        cls_output = self.classifier(cls_output)  # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = Focal_Loss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output


class TransposeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(x):
        return torch.transpose(x, [0, 2, 1]).contiguous()


class BertClassifierCustom(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert_config = AutoConfig.from_pretrained(args.bertname)
        if os.path.exists(args.bert_path):
            with open(args.bert_path) as f:
                self.bert = torch.load(f)
        else:
            bert = BertModel.from_pretrained(args.bertname)
            # remove the pooling layer at the end of bertmodel
            self.bert = torch.nn.Sequential(*(list(bert.children())[:-1]))
            with open(args.bert_path) as f:
                torch.save(self.bert, f)

        if args.classifier == 'cnn':
            self.classifier = self.build_cnn_layer(bert_config.hidden_size, args)
        elif args.classifier == 'bi_lstm':
            self.classifier = self.build_bi_lstm_layer(bert_config.hidden_size, args)
        elif args.classifier == 'capsule':
            self.classifier = self.build_capsule_net(bert_config.hidden_size)
        else:
            raise NotImplementedError

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        cls_output = self.classifier(outputs)  # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = nn.BCELoss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output

    def build_cnn_layer(self, input_dim, args):
        cnn_output_len1 = (512 - args.cnn_kernel) // args.stride + 1
        cnn_output_len2 = (cnn_output_len1 - args.kernel) // args.stride + 1
        return nn.Sequential(
            TransposeModule(),
            nn.Conv1d(input_dim, args.cnn_dim, args.cnn_kernel, stride=args.cnn_stride),
            nn.LayerNorm([args.cnn_dim, cnn_output_len1]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(cnn_output_len2 * args.cnn_dim, args.num_classes),
        )

    def build_bi_lstm_layer(self, input_dim, args):
        return nn.Sequential(
            nn.LSTM(input_dim, args.lstm_hidden_size // 2,
                    num_layers=1, bidirectional=True, batch_first=True),
            nn.LayerNorm([]),
            nn.Tanh(),
            nn.Flatten(),
            nn.Linear(512 * args.lstm_hidden_size, args.num_classes),
        )

    def build_capsule_net(self, input_dim, args):
        return nn.Sequential(
            TransposeModule(),
            Capsule(input_dim=input_dim, num_capsule=args.num_capsule,
                    dim_capsule=args.dim_capsule, routings=args.routings, kernel_size=(args.capsule_kernel, 1)),
            nn.Flatten(),
            nn.Linear(self.num_capsule * self.dim_capsule, args.num_classes),
        )