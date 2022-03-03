from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
from utils import Focal_Loss

def load_tokenizer(bert_model_name):
    tokenizer = BertTokenizer.from_pretrained(bert_model_name, do_lower_case=True)
    assert tokenizer.pad_token_id == 0, "Padding value used in masks is set to zero, please change it everywhere"
    return tokenizer


class BertClassifier(nn.Module):
    def __init__(self, bert: BertModel, num_classes: int):
        super().__init__()
        self.bert = bert
        self.classifier = nn.Linear(bert.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
            labels=None):
        outputs = self.bert(input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids,
                               position_ids=position_ids,
                               head_mask=head_mask)
        cls_output = outputs[1] # batch, hidden
        cls_output = self.classifier(cls_output) # batch, 6
        cls_output = torch.sigmoid(cls_output)
        criterion = Focal_Loss()
        loss = 0
        if labels is not None:
            loss = criterion(cls_output, labels)
        return loss, cls_output

