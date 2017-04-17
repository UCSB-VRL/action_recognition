import os, time, sys
import numpy as np
import argparse
import datetime

import torch.nn as nn
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from model import lstm
from dataset import data
from utils import add_summary, adjust_learning_rate, mean_average_precision
from utils import accuracy
from utils import AverageMeter

import tensorflow as tf

parser = argparse.ArgumentParser(description='Train or eval action recognition through LSTMs')
parser.add_argument('--test-splitfile', action="store", dest='test_splitfile',
                    help="Location the test split of dataset is specified")
parser.add_argument('--train-splitfile', action="store", dest='train_splitfile',
                    help="Location the train split of dataset is specified")
parser.add_argument('--class-indexfile', action="store", dest='class_indexfile',
                    help="Location the class name to index map is stored")
parser.add_argument('--sequence-length', action="store", dest='sequence_length',
                    help="Number of frames modelled at a time", default=16,
                    type=int)
parser.add_argument('--batch-size', action="store", dest='batch_size',
                    help="Batch size", default=32,
                    type=int)
parser.add_argument('--feature-extractortype', action="store", dest='feature_extractortype',
                    default='vgg16', help="Type of feature extractor used on images")
parser.add_argument('--nhid', type=int, default=200, dest='nhid',
                    help='number of hidden units per layer in LSTM')
parser.add_argument('--nlayers', type=int, default=1, dest='nlayers',
                    help='number of layers in LSTM')
parser.add_argument('--log-interval', type=int, default=20, dest='log_interval',
                    help='report interval')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--feature-root', default='', dest='feature_root',
                    help='Pre-computed features are saved under location feature_root"')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum',
                    help='momentum weight')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    dest='weight_decay', help='weight decay (default: 1e-4)')
parser.add_argument('--tfboard-dir',default='', type=str,
                    dest='tfboard_dir', help='Location where tensorboard plotting and trained model data is stored')
parser.add_argument('--dropout', default=0.0, type=float, dest='dropout',
                    help='dropout factor for encoder, decoder, lstm modules')
parser.add_argument('--run-name', default='', type=str, dest='run_name',
                    help='Run name')
parser.add_argument('-g', '--gpu', default='0', type=str, dest='gpu',
                    help='selected gpu id for current run')                    
parser.add_argument('-j', '--workers', default=6, type=int, dest='workers',
                    help='number of data loading workers (default: 6)')
parser.add_argument('--dataset-name', default='ucf101', dest='dataset_name',
                    help='The dataset name"')

args = parser.parse_args()
print('=' * 89)
print 'Command Line Arguments'
for x in args.__dict__.keys():
  print str(x) + ' : ' + str(args.__dict__[x])
print('=' * 89)

global_step = 0

# tensorboard structures
if args.run_name == '':
  foldername = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')
else:
  foldername = args.run_name + '_' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

tf_summary_folder = os.path.join(args.tfboard_dir, foldername)
tf_summary_writer = tf.summary.FileWriter(tf_summary_folder)

feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }
if args.dataset_name == 'ucf101':
  num_classes = 101
elif args.dataset_name == 'hmdb51':
  num_classes = 51
else:
  assert 0, 'Unexpected dataset-name : %s'%args.dataset_name


os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

lstm_model = lstm.LSTMModel(feat_size_map[args.feature_extractortype], 
                           args.nhid, args.nlayers,
                           num_classes, 
                           args.dropout)
lstm_model.cuda()

# define loss function (criterion) and optimizer
lstm_criterion = nn.CrossEntropyLoss().cuda()
valid_criterion = nn.NLLLoss().cuda()
valid_func = nn.Softmax().cuda()

lstm_optimizer = torch.optim.SGD(lstm_model.parameters(), args.lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)
cudnn.benchmark = True

ids_file = args.class_indexfile
class2idx = data.class_dict(ids_file)

test_file = args.test_splitfile
test_vids = data.video_list(test_file, class2idx)

train_file = args.train_splitfile
train_vids = data.video_list(train_file, class2idx)
train_vids, valid_vids = data.train_split(train_vids)

print(len(train_vids), len(valid_vids), len(test_vids))

root = args.feature_root
# Data loading
train_loader = DataLoader(data.VideoList(root, train_vids, for_train=True, 
                                             seq_length=args.sequence_length),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True,
                              collate_fn=data.collate)

valid_loader = DataLoader(data.VideoList(root, valid_vids, for_train=False, 
                                         seq_length=args.sequence_length),
                          batch_size=args.batch_size, shuffle=False,
                          num_workers=args.workers, pin_memory=True,
                          collate_fn=data.collate)

test_loader = DataLoader(data.VideoList(root, test_vids, for_train=False,
                                        seq_length=args.sequence_length),
                         batch_size=args.batch_size, shuffle=False,
                         num_workers=args.workers, pin_memory=True,
                         collate_fn=data.collate)

valid_targets = np.array([v[1] for v in valid_vids])
test_targets  = np.array([v[1] for v in test_vids])



def train(train_loader, model, criterion, optimizer, epoch):
  global global_step
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to train mode
  model.train()
  batch_t0 = time.time()
  num_batches = 0
  for i, (input, target, video_names, frame_indexes) in enumerate(train_loader):
    num_batches += 1

    input = input.cuda(async=True)
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input)
    target_var = torch.autograd.Variable(target)

    # compute output
    output = model(input_var)
    loss = criterion(output, target_var)

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1,5))
    losses.update(loss.data[0], target.size(0))
    top1.update(prec1[0], target.size(0))
    top5.update(prec5[0], target.size(0))

    # compute gradient and do SGD step
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if global_step%args.log_interval == 0:
      add_summary(tf_summary_writer, tag="Train_Prec1", raw_value=prec1[0], global_step=global_step)
      add_summary(tf_summary_writer, tag="Train_Prec5", raw_value=prec5[0], global_step=global_step)
      add_summary(tf_summary_writer, tag="Train_Loss", raw_value=loss.data[0], global_step=global_step)
      add_summary(tf_summary_writer, tag="Learning_Rate", raw_value=lr, global_step=global_step)

      elapsed_time = str(datetime.timedelta(seconds=time.time()-zero_time))
      batch_time = time.time() - batch_t0 
      batch_t0 = time.time()
      ms_per_batch = 1000*(batch_time/num_batches); num_batches = 0
      print 'elapsed_time : {} | epoch {:3d} | {:6d}/{:6d} batches | lr {:02.6f} | ms/batch {:8.2f} | ' \
            'loss {:5.2f} | top1 {:8.2f} | top5 {:8.2f}'.format(elapsed_time, 
                                                                epoch, 
                                                                i, 
                                                                len(train_loader),
                                                                lr,
                                                                ms_per_batch,
                                                                loss.data[0],
                                                                prec1[0],
                                                                prec5[0])
                              
    global_step += 1

  return losses.avg, top1.avg, top5.avg 

def validate(valid_loader, model, criterion, func, targets):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()

  # switch to evaluate mode
  model.eval()

  outputs = []
  for i, (input, target, video_names, frame_indexes) in enumerate(valid_loader):
    b = target.size(0)
    input = input.cuda(async=True)
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    
    # compute output
    output = model(input_var)
    output = func(output)     # softmax, (bxr)x101
    
    # mean probabilities over r sequences
    output = output.view(b, -1, num_classes).mean(1).squeeze(1)
    loss = criterion(torch.log(output), target_var) # needs logsoftmax
    outputs.append(output.data.cpu().numpy())

    # measure accuracy and record loss
    prec1, prec5 = accuracy(output.data, target, topk=(1,5))
    losses.update(loss.data[0], target.size(0))
    top1.update(prec1[0], target.size(0))
    top5.update(prec5[0], target.size(0))

  outputs = np.concatenate(outputs, axis=0)
  mean_ap = mean_average_precision(outputs, targets, num_classes)

  return losses.avg, top1.avg, top5.avg, mean_ap


zero_time = time.time()
print '-------------- New training session ----------------'
for epoch in range(0, args.epochs):
  epoch_start_time = time.time()

  lr = adjust_learning_rate(lstm_optimizer, epoch, args.lr)

  # train_loss, train_prec1
  train_scores = train(train_loader, lstm_model, lstm_criterion, 
                        lstm_optimizer, epoch)

  if (epoch+1) % 1 == 0:
    # valid_loss, valid_prec1, valid_map
    valid_scores = validate(valid_loader, lstm_model, valid_criterion, 
                            valid_func, valid_targets)  
    add_summary(tf_summary_writer, tag="Val_Loss", raw_value=valid_scores[0], global_step=global_step)
    add_summary(tf_summary_writer, tag="Val_Prec1", raw_value=valid_scores[1], global_step=global_step)
    add_summary(tf_summary_writer, tag="Val_Prec5", raw_value=valid_scores[2], global_step=global_step)
    add_summary(tf_summary_writer, tag="Val_MAP", raw_value=valid_scores[3], global_step=global_step)

    print('-' * 89)
    print('| VAL : end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'top1 {:8.2f} | top5 {:8.2f} | MAP {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        valid_scores[0], valid_scores[1], valid_scores[2], valid_scores[3]))
    print('-' * 89)

    # test_loss, test_prec1, test_map
    test_scores = validate(test_loader, lstm_model, valid_criterion, 
                            valid_func, test_targets)
    add_summary(tf_summary_writer, tag="Test_Loss", raw_value=test_scores[0], global_step=global_step)
    add_summary(tf_summary_writer, tag="Test_Prec1", raw_value=test_scores[1], global_step=global_step)
    add_summary(tf_summary_writer, tag="Test_Prec5", raw_value=test_scores[2], global_step=global_step)
    add_summary(tf_summary_writer, tag="Test_MAP", raw_value=test_scores[3], global_step=global_step)

    print('-' * 89)
    print('| TEST : end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'top1 {:8.2f} | top5 {:8.2f} | MAP {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                        test_scores[0], test_scores[1], test_scores[2], test_scores[3]))
    print('-' * 89)

    scores = train_scores + valid_scores + test_scores

    # remember evaluation results and save checkpoint
    out_path = os.path.join(tf_summary_folder, 'model_epoch_%d.tar' % (epoch + 1, ))
    torch.save({
        'epoch': epoch + 1,
        'state_dict': lstm_model.state_dict(),
        'optim_dict': lstm_optimizer.state_dict(),
        'evaluations': scores,
        'nhid': args.nhid,
        'sequence_length': args.sequence_length,
        'nlayers': args.nlayers,
        'dataset_name': args.dataset_name,
        'feature_extractortype': args.feature_extractortype,
        'dropout': args.dropout
    }, out_path)




