from __future__ import print_function

import time
import gc
import os
import argparse

import numpy as np
from sklearn.externals import  joblib
import torch
from torch import nn
import torch.backends.cudnn as cudnn

#from vocab import  VocabBuilder, GloveVocabBuilder
from vocab import  VocabBuilder
from dataloader import TextClassDataLoader, TextClassDataLoader_multi
from model import RNN
from util import AverageMeter, accuracy, accuracy_thresh
from util import adjust_learning_rate
import ipdb
from glob import glob

np.random.seed(0)
torch.manual_seed(0)

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--epochs', default=50, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=0.005, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', '-p', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--save-freq', '-sf', default=10, type=int, metavar='N', help='model save frequency(epoch)')
parser.add_argument('--embedding-size', default=50, type=int, metavar='N', help='embedding size')
parser.add_argument('--hidden-size', default=128, type=int, metavar='N', help='rnn hidden size')
parser.add_argument('--layers', default=2, type=int, metavar='N', help='number of rnn layers')
parser.add_argument('--classes', default=8, type=int, metavar='N', help='number of output classes')
parser.add_argument('--min-samples', default=5, type=int, metavar='N', help='min number of tokens')
parser.add_argument('--cuda', default=False, action='store_true', help='use cuda')
parser.add_argument('--glove', default='glove/glove.6B.100d.txt', help='path to glove txt')
parser.add_argument('--data', default='data', help='path to data')
parser.add_argument('--rnn', default='LSTM', choices=['LSTM', 'GRU'], help='rnn module type')
parser.add_argument('--mean_seq', default=False, action='store_true', help='use mean of rnn output')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('-m','--model', default='default', help='path to checkpoint')
parser.add_argument('--multi_label', default=False, action='store_true', help='multi label classification')
args = parser.parse_args()



# create vocab
print("===> creating vocabs ...")
end = time.time()
v_builder, d_word_index, embed = None, None, None
#if os.path.exists(args.glove):
#    v_builder = GloveVocabBuilder(path_glove=args.glove)
#    d_word_index, embed = v_builder.get_word_index()
#    args.embedding_size = embed.size(1)
#else:
sample = glob(args.data+'/*')[0]
if sample.endswith('csv'):
    train_file = os.path.join(args.data, 'trainval.csv')
    val_file = os.path.join(args.data, 'val.csv')
    test_file = os.path.join(args.data, 'test.csv')
else:
    train_file = os.path.join(args.data, 'trainval.tsv')
    val_file = os.path.join(args.data, 'val.tsv')
    test_file = os.path.join(args.data, 'test.tsv')
    
v_builder = VocabBuilder(path_file=train_file)
d_word_index, embed = v_builder.get_word_index(min_sample=args.min_samples)

model_dir = os.path.join('checkpoints', args.model)
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)
joblib.dump(d_word_index, os.path.join(model_dir, 'd_word_index.pkl'), compress=3)
print('===> vocab creatin: {t:.3f}'.format(t=time.time()-end))

print('args: ',args)

# create trainer
print("===> creating dataloaders ...")
end = time.time()
if not args.multi_label:
    train_loader = TextClassDataLoader(train_file, d_word_index, batch_size=args.batch_size)
    val_loader = TextClassDataLoader(val_file, d_word_index, batch_size=args.batch_size)
    test_loader = TextClassDataLoader(test_file, d_word_index, batch_size=args.batch_size)
else:
    train_loader = TextClassDataLoader_multi(train_file, d_word_index, batch_size=args.batch_size)
    val_loader = TextClassDataLoader_multi(val_file, d_word_index, batch_size=args.batch_size)
    test_loader = TextClassDataLoader_multi(test_file, d_word_index, batch_size=args.batch_size)
print('===> dataloader creatin: {t:.3f}'.format(t=time.time()-end))


# create model
print("===> creating rnn model ...")
vocab_size = len(d_word_index)
model = RNN(vocab_size=vocab_size, embed_size=args.embedding_size, num_output=args.classes, rnn_model=args.rnn,
            use_last=( not args.mean_seq),
            hidden_size=args.hidden_size, embedding_tensor=embed, num_layers=args.layers, batch_first=True)
print(model)

# optimizer and loss
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay)

if args.multi_label:
    criterion = nn.BCEWithLogitsLoss()
else:
    criterion = nn.CrossEntropyLoss()

print(optimizer)
print(criterion)

if args.cuda:
    torch.backends.cudnn.enabled = True
    cudnn.benchmark = True
    model.cuda()
    criterion = criterion.cuda()


def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target, seq_lengths) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
        
        # compute output
        output = model(input, seq_lengths)
        loss = criterion(output, target)
        
        # measure accuracy and record loss
        if args.multi_label:
            prec1 = accuracy_thresh(output.data, target)
            top1.update(prec1, input.size(0))
        else:
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0][0], input.size(0))
        losses.update(loss.data, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i != 0 and i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})  Loss {loss.val:.4f} ({loss.avg:.4f})  '
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, i, len(train_loader), batch_time=batch_time, data_time=data_time, loss=losses, top1=top1))
            gc.collect()


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target,seq_lengths) in enumerate(val_loader):

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input,seq_lengths)
        loss = criterion(output, target)

        if args.multi_label:
            prec1 = accuracy_thresh(output.data, target)
            top1.update(prec1, input.size(0))
        else:
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0][0], input.size(0))
        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        #top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if i!= 0 and i % args.print_freq == 0:
            print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
                  'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
            gc.collect()

    print(' DEV Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


def test(test_loader, model, criterion):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    end = time.time()
    for i, (input, target,seq_lengths) in enumerate(val_loader):

        if args.cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        # compute output
        output = model(input,seq_lengths)
        loss = criterion(output, target)

        if args.multi_label:
            prec1 = accuracy_thresh(output.data, target)
            top1.update(prec1, input.size(0))
        else:
            prec1 = accuracy(output.data, target, topk=(1,))
            top1.update(prec1[0][0], input.size(0))
        # measure accuracy and record loss
        #prec1 = accuracy(output.data, target, topk=(1,))
        losses.update(loss.data, input.size(0))
        #top1.update(prec1[0][0], input.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        #if i!= 0 and i % args.print_freq == 0:
        #    print('Test: [{0}/{1}]  Time {batch_time.val:.3f} ({batch_time.avg:.3f})  '
        #          'Loss {loss.val:.4f} ({loss.avg:.4f})  Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
        #           i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1))
        #    gc.collect()

    print(' TEST Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg


# training and testing
for epoch in range(1, args.epochs+1):

    adjust_learning_rate(args.lr, optimizer, epoch)
    train(train_loader, model, criterion, optimizer, epoch)
    validate(val_loader, model, criterion)

    # save current model
    if epoch % args.save_freq == 0:
        name_model = 'rnn_{}.pkl'.format(epoch)
        path_save_model = os.path.join(model_dir, name_model)
        joblib.dump(model.float(), path_save_model, compress=2)

print('testing...')
test(test_loader, model, criterion)
