import os
import pandas as pd
import numpy as np
import argparse
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from functools import partial
from os.path import join as pjoin
from torch.utils.data import DataLoader, RandomSampler
from data import ToxicDataset, load_data, collate_fn
from model import BertClassifier, BertClassifierCustom
from utils import *

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def init(args):
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)


def save_checkpoint(model, suffix):
    torch.save(model.state_dict(), pjoin('./checkpoints', suffix + '.pth'))


def train(model, criterion, iterator, optimizer, scheduler):
    model.train()
    total_loss = 0
    for x, y in iterator:
        optimizer.zero_grad()
        mask = (x != 0).float()
        outputs = model(x, attention_mask=mask)
        loss = criterion(outputs, y)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        scheduler.step()
    print(f"Train loss {total_loss / len(iterator)}")


def evaluate(model, criterion, iterator):
    model.eval()
    pred = []
    true = []
    with torch.no_grad():
        total_loss = 0
        for x, y in iterator:
            mask = (x != 0).float()
            outputs = model(x, attention_mask=mask, labels=y)
            loss = criterion(outputs, y)
            total_loss += loss
            true += y.cpu().numpy().tolist()
            pred += outputs.cpu().numpy().tolist()
    true = np.array(true)
    pred = np.array(pred)
    for i, name in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        print(f"{name} roc_auc {roc_auc_score(true[:, i], pred[:, i])}")
    print(f"Evaluate loss {total_loss / len(iterator)}")


def test(args, tokenizer, model):
    model.eval()

    test_df = pd.read_csv(os.path.join(args.path, 'test.csv'))
    submission = pd.read_csv(os.path.join(args.path, 'sample_submission.csv'))
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    for i in range(len(test_df) // args.batch_size + 1):
        batch_df = test_df.iloc[i * args.batch_size: (i + 1) * args.batch_size]
        assert (batch_df["id"] == submission["id"][i * args.batch_size: (i + 1) * args.batch_size]).all(), f"Id mismatch"
        texts = []
        for text in batch_df["comment_text"].tolist():
            text = tokenizer.encode(text, add_special_tokens=True, max_length=128, truncation=True)
            texts.append(torch.LongTensor(text))
        x = pad_sequence(texts, batch_first=True, padding_value=tokenizer.pad_token_id).to(device)
        mask = (x != tokenizer.pad_token_id).float().to(device)
        with torch.no_grad():
            outputs = model(x, attention_mask=mask)
        outputs = outputs.cpu().numpy()
        submission.iloc[i * args.batch_size: (i + 1) * args.batch_size][columns] = outputs

    save_dir_number = 0
    save_dir = pjoin(args.output_dir, f"submission_{args.exp_name}_0")
    while os.path.exists(save_dir + '.csv'):
        save_dir_number += 1
        save_dir_splits = save_dir.split('_')
        save_dir = '_'.join(save_dir_splits[:-1] + [str(save_dir_number)])
    submission.to_csv(save_dir + '.csv', index=False)

    if os.path.exists(args.true_label_csv):
        local_auc = local_evaluate(submission, args.true_label_csv, columns)
        print('Column-wise AUC: ')
        print(' | '.join(columns + ['average']))
        print(' | '.join(map(str, local_auc)))


def load_iterators(args, tokenizer, columns=None):
    train_df, val_df = load_data(args.path)
    train_dataset = ToxicDataset(tokenizer, train_df, args.max_len, lazy=True, columns=columns)
    dev_dataset = ToxicDataset(tokenizer, val_df, args.max_len, lazy=True, columns=columns)
    collate_fn_m = partial(collate_fn, device=device)
    train_sampler = RandomSampler(train_dataset)
    dev_sampler = RandomSampler(dev_dataset)
    train_iterator = DataLoader(train_dataset, batch_size=args.batch_size,
                                sampler=train_sampler, collate_fn=collate_fn_m)
    dev_iterator = DataLoader(dev_dataset, batch_size=args.batch_size, sampler=dev_sampler, collate_fn=collate_fn_m)
    return train_iterator, dev_iterator


def train_and_eval(args):
    tokenizer = load_tokenizer(args.bertname)
    train_iterator, dev_iterator = load_iterators(args, tokenizer)

    if args.classifier == 'none':
        model = BertClassifier(args).to(device)
    else:
        model = BertClassifierCustom(args).to(device)

    criterion = Focal_Loss(alpha=args.alpha) if args.focal_loss else nn.BCELoss()

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays
    total_steps = len(train_iterator) * args.epochs - args.warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    for i in range(args.epochs):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, criterion, train_iterator, optimizer, scheduler)
        evaluate(model, criterion, dev_iterator)

    save_checkpoint(model, args.exp_name)
    test(args, tokenizer, model)


def train_and_eval_teacher(args):
    tokenizer = load_tokenizer(args.bertname)
    train_iterator, dev_iterator = load_iterators(args, tokenizer)

    if args.classifier == 'none':
        model = BertClassifier(args).to(device)
    else:
        model = BertClassifierCustom(args).to(device)

    if args.resume:
        print('Fine-tune model from ' + args.resume)
        state_dicts = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dicts)

    tasks = np.array(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate'])
    teacher_task_index = np.where(tasks == args.task)[0][0]
    criterion = Focal_Loss_Teacher(teacher_task_index, alpha=args.alpha)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays
    total_steps = len(train_iterator) * args.epochs - args.warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    for i in range(args.epochs):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, criterion, train_iterator, optimizer, scheduler)
        evaluate(model, criterion, dev_iterator)
        test(args, tokenizer, model)

    save_checkpoint(model, args.exp_name)


def train_and_eval_student(args):
    columns = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
    columns += list(map(lambda x: x + args.suffix, columns))
    tokenizer = load_tokenizer(args.bertname)
    train_iterator, dev_iterator = load_iterators(args, tokenizer, columns=columns)

    if args.classifier == 'none':
        model = BertClassifier(args).to(device)
    else:
        model = BertClassifierCustom(args).to(device)

    if args.resume:
        print('Fine-tune model from ' + args.resume)
        state_dicts = torch.load(args.resume, map_location=device)
        model.load_state_dict(state_dicts)

    criterion = Focal_Loss_Distill(alpha=args.alpha)

    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # triangular learning rate, linearly grows untill half of first epoch, then linearly decays
    total_steps = len(train_iterator) * args.epochs - args.warmup_steps
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, total_steps)

    for i in range(args.epochs):
        print('=' * 50, f"EPOCH {i}", '=' * 50)
        train(model, criterion, train_iterator, optimizer, scheduler)
        evaluate(model, criterion, dev_iterator)
        test(args, tokenizer, model)

    save_checkpoint(model, args.exp_name)


if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Toxic Comments Classification')
    parser.add_argument('--exp_name', type=str, default='exp')
    parser.add_argument('--datapath', type=str, default='../data')
    parser.add_argument('--bertname', type=str, default='bert-base-uncased')
    parser.add_argument('--path', type=str, default='./')
    parser.add_argument('--max_len', type=int, default=128)
    parser.add_argument('--seed', type=int, default=403)
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--true_label_csv', type=str, default='./true_labels.csv')
    parser.add_argument('--output_dir', type=str, default='./')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--warmup_steps', type=int, default=10**3)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--focal_loss', action='store_true')
    parser.add_argument('--num_classes', type=int, default=6)
    parser.add_argument('--bert_path', type=str, default='./bert_trunct.pkl')
    parser.add_argument('--lstm_hidden_size', type=int, default=20)
    parser.add_argument('--cnn_dim', type=int, default=20)
    parser.add_argument('--cnn_kernel', type=int, default=3)
    parser.add_argument('--cnn_stride', type=int, default=2)
    parser.add_argument('--num_capsule', type=int, default=10)
    parser.add_argument('--dim_capsule', type=int, default=16)
    parser.add_argument('--routings', type=int, default=3)
    parser.add_argument('--capsule_kernel', type=int, default=9)
    parser.add_argument('--classifier', type=str, choices=['bi_lstm', 'cnn', 'capsule', 'none'], default='none')
    parser.add_argument('-t', '--task', type=str,
                        choices=['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate', 'none'],
                        default='none')
    parser.add_argument('--kd_flag', type=str, choices=['teacher', 'student', 'none'], default='none')
    parser.add_argument('--alpha', type=float, default=0.5)
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--suffix', type=str, default='_ft', help='suffix of teacher labels for distillation')

    args = parser.parse_args()
    print(args)

    init(args)

    if args.kd_flag == 'teacher':
        train_and_eval_teacher(args)
    elif args.kd_flag == 'student':
        train_and_eval_student(args)
    else:
        train_and_eval(args)
