import os, sys
import numpy as np
import argparse

import torch.nn as nn
import torch
from torch.utils.data import DataLoader

from model import lstm
from dataset import data
from utils import mean_average_precision
from utils import conf_matrix, plot_confusion_matrix, accuracy
from utils import AverageMeter

parser = argparse.ArgumentParser(description='Examine the activation stats for trained LSTM models')
parser.add_argument('--test-splitfile', action="store", dest='test_splitfile',
                    help="Location the test split of dataset is specified")
parser.add_argument('--train-splitfile', action="store", dest='train_splitfile',
                    help="Location the train split of dataset is specified")
parser.add_argument('--class-indexfile', action="store", dest='class_indexfile',
                    help="Location the class name to index map is stored")
parser.add_argument('--batch-size', action="store", dest='batch_size',
                    help="Batch size", default=32, type=int)
parser.add_argument('--feature-root', default='', dest='feature_root',
                    help='Pre-computed features are saved under location feature_root"')
parser.add_argument('--model-path',default='', type=str,
                    dest='model_path', help='Location from where lstm model is to be loaded')
parser.add_argument('-g', '--gpu', default='0', type=str, dest='gpu',
                    help='selected gpu id for current run')                    
parser.add_argument('-j', '--workers', default=6, type=int, dest='workers',
                    help='number of data loading workers (default: 6)')

args = parser.parse_args()
print('=' * 89)
print 'Command Line Arguments'
for x in args.__dict__.keys():
  print str(x) + ' : ' + str(args.__dict__[x])
print('=' * 89)

if args.model_path:
  if os.path.isfile(args.model_path):
      print("=> loading checkpoint '{}'".format(args.model_path))
      checkpoint = torch.load(args.model_path)
      args.start_epoch = checkpoint['epoch']
      args.feature_extractortype = checkpoint['feature_extractortype']
      args.dataset_name = checkpoint['dataset_name']
      args.nhid = checkpoint['nhid']
      args.dropout = checkpoint['dropout']
      args.sequence_length = checkpoint['sequence_length']
      args.nlayers = checkpoint['nlayers']
      print("=> loaded checkpoint '{}' (epoch {})"
            .format(args.model_path, checkpoint['epoch']))
  else:
      print("=> no checkpoint found at '{}'".format(args.model_path))
      sys.exit()
else:
  print("=> no checkpoint passed")
  sys.exit()

if args.dataset_name == 'ucf101':
  num_classes = 101
elif args.dataset_name == 'hmdb51':
  num_classes = 51
else:
  assert 0, 'Unexpected dataset-name : %s'%args.dataset_name

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }

lstm_model = lstm.LSTMModel(feat_size_map[args.feature_extractortype], 
                           args.nhid, args.nlayers,
                           num_classes, 
                           args.dropout)
lstm_model.cuda()
print("=> loading lstm weights")
lstm_model.load_state_dict(checkpoint['state_dict'])
print("=> loaded lstm weights")

valid_criterion = nn.NLLLoss().cuda()
valid_func = nn.Softmax().cuda()

ids_file = args.class_indexfile
class2idx = data.class_dict(ids_file)
classes = class2idx.keys()
classes.sort()

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

def evaluate(eval_loader, model, criterion, func, targets):
  losses = AverageMeter()
  top1 = AverageMeter()
  top5 = AverageMeter()
  
  # switch to evaluate mode
  model.eval()

  outputs = []
  for i, (input, target, video_name, frame_indexes) in enumerate(eval_loader):
    b = target.size(0)
    input = input.cuda(async=True)
    target = target.cuda(async=True)
    input_var = torch.autograd.Variable(input, volatile=True)
    target_var = torch.autograd.Variable(target, volatile=True)
    
    # compute output
    output  = model(input_var)
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
  conf_mat = conf_matrix(outputs,targets)

  return losses.avg, top1.avg, top5.avg, mean_ap, conf_mat

results = evaluate(test_loader, lstm_model, valid_criterion, valid_func, test_targets)

print('-' * 89)
print('| VAL   loss {:5.2f} | '
        'top1 {:8.2f} | top5 {:8.2f} | MAP {:8.2f}'.format( results[0], 
                                                            results[1],
                                                            results[2],
                                                            results[3]))
print('-' * 89)

conf_mat = results[4]
plot_confusion_matrix(conf_mat, classes, normalize=True, image_name='confusion_matrix_normed.png')
for c_idx, c in enumerate(classes):
  conf_classes = np.where(conf_mat[c_idx,:] != 0)[0] # confused classes
  print '%s %02d  || '%(c, conf_mat[c_idx, c_idx]),
  for x in conf_classes:
    if  x != c_idx:
      print '%s %02d '%(classes[x], conf_mat[c_idx, x]),

  print ''

import ipdb; ipdb.set_trace()