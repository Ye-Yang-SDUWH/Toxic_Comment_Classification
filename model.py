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
        return cls_output


class TransposeModule(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.permute([0, 2, 1]).contiguous()


class BiLSTM(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.layer = nn.LSTM(input_dim, output_dim // 2,
                             num_layers=1, bidirectional=True, batch_first=True)

    def forward(self, x):
        return self.layer(x)[0]


class BertClassifierCustom(nn.Module):
    def __init__(self, args):
        super().__init__()
        bert_config = AutoConfig.from_pretrained(args.bertname)
        self.bert = BertModel.from_pretrained(args.bertname)

        if args.classifier == 'cnn':
            self.classifier = self.build_cnn_layer(bert_config.hidden_size, args)
            self.capsule = self.build_capsule_net(args.cnn_dim, args)
        elif args.classifier == 'bi_lstm':
            self.classifier = self.build_bi_lstm_layer(bert_config.hidden_size, args)
            self.capsule = self.build_capsule_net(args.lstm_hidden_size, args)
        else:
            self.classifier = nn.Identity()
            self.capsule = self.build_capsule_net(bert_config.hidden_size, args)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                labels=None):
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask).last_hidden_state
        # outputs : [N, L, C]
        cls_output = self.capsule(self.classifier(outputs))  # batch, 6
        cls_output = torch.sigmoid(cls_output)
        return cls_output

    def build_cnn_layer(self, input_dim, args):
        return nn.Sequential(
            TransposeModule(),
            nn.Conv1d(input_dim, args.cnn_dim, args.cnn_kernel, stride=args.cnn_stride),
            nn.Dropout(0.2),
            nn.LeakyReLU(),
            TransposeModule()
        )

    def build_bi_lstm_layer(self, input_dim, args):
        return nn.Sequential(
            BiLSTM(input_dim, args.lstm_hidden_size),
            nn.Dropout(0.2),
            nn.Tanh()
        )

    def build_capsule_net(self, input_dim, args):
        return nn.Sequential(
            TransposeModule(),
            Capsule(input_dim=input_dim, num_capsule=args.num_capsule,
                    dim_capsule=args.dim_capsule, routings=args.routings, kernel_size=(args.capsule_kernel, 1)),
            nn.Flatten(),
            nn.Linear(args.num_capsule * args.dim_capsule, args.num_classes)
        )
