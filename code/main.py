import os, sys, time
import numpy as np
from model import  cnn
from model import lstm
from dataset import ucf101_dataset
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch
import math
import datetime

parser = argparse.ArgumentParser(description='Train or eval action recognition through LSTMs')
parser.add_argument('--dataset_root', action="store", dest='dataset_root',
                    help="Location where images extracted from the UCF-101 dataset are stored")
parser.add_argument('--test_splitfile', action="store", dest='test_splitfile',
                    help="Location the test split of dataset is specified")
parser.add_argument('--train_splitfile', action="store", dest='train_splitfile',
                    help="Location the train split of dataset is specified")
parser.add_argument('--class_indexfile', action="store", dest='class_indexfile',
                    help="Location the class name to index map is stored")
parser.add_argument('--sequence_length', action="store", dest='sequence_length',
                    help="Number of frames modelled at a time", default=16,
                    type=int)
parser.add_argument('--batch_size', action="store", dest='batch_size',
                    help="Batch size", default=32,
                    type=int)
parser.add_argument('--feature_extractortype', action="store", dest='feature_extractortype',
                    default='vgg16', help="Type of feature extractor used on images")
parser.add_argument('--nhid', type=int, default=200, dest='nhid',
                    help='number of hidden units per layer in LSTM')
parser.add_argument('--nlayers', type=int, default=1, dest='nlayers',
                    help='number of layers in LSTM')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=20, dest='log_interval',
                    help='report interval')
parser.add_argument('--lr', type=float, default=0.01,
                    help='initial learning rate')
parser.add_argument('--epochs', type=int, default=20,
                    help='upper epoch limit')
parser.add_argument('--clip', type=float, default=0.5,
                    help='gradient clipping')

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)


feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }


dataset = ucf101_dataset.ucf101Dataset(data_root=args.dataset_root, test_splitfile=args.test_splitfile,
                                        train_splitfile=args.train_splitfile, class_indfile=args.class_indexfile,
                                       seq_length=args.sequence_length, batch_size=args.batch_size)
dataset.load_splits()
dataset.create_seq_pkl(split_type='train')
dataset.create_seq_pkl(split_type='val')
dataset.create_seq_pkl(split_type='test')

train_data = dataset.load_sequence_data(split_type='train')
test_data = dataset.load_sequence_data(split_type='test')
val_data = dataset.load_sequence_data(split_type='val')

feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype, batch_size=args.batch_size)
feature_cnn = feature_extractor.create_cnn()

traindata_loader = feature_extractor.create_dataloader(train_data, split_type='train')
testdata_loader = feature_extractor.create_dataloader(test_data, split_type='test')
valdata_loader = feature_extractor.create_dataloader(val_data, split_type='val')

lstm_model = lstm.RNNModel('LSTM', feat_size_map[args.feature_extractortype], args.nhid, args.nlayers, dataset.get_numclasses())
lstm_model.cuda()

lstm_criterion = nn.CrossEntropyLoss()

def repackage_hidden(h):
  """Wraps hidden states in new Variables, to detach them from their history."""
  if type(h) == Variable:
    return Variable(h.data)
  else:
    return tuple(repackage_hidden(v) for v in h)

def train():
  total_loss = 0
  start_time = time.time()
  zero_batch_time = time.time()
  hidden = lstm_model.init_hidden(args.batch_size)
  for batch_idx, (train_seq, train_label) in enumerate(traindata_loader):
    num_batchsamples = len(train_seq[0])  # Hack, to avoid running into incomplete batches
    if num_batchsamples < args.batch_size:
      continue
    image_features = []
    lstm_label = Variable(torch.cat([train_label.unsqueeze(0) ]*args.sequence_length).view(-1).cuda())
    for image in train_seq:
        image = Variable(image.cuda(), volatile=True)
        image_features.append(feature_cnn(image).squeeze().unsqueeze(0))

    lstm_input =  torch.cat(image_features) # seq_length x batch_size x feature_size
    lstm_input = Variable(lstm_input.data.cuda(), volatile=False)

    hidden = repackage_hidden(hidden)
    lstm_model.zero_grad()
    output, hidden = lstm_model(lstm_input, hidden)
    loss = lstm_criterion(output.view(-1, dataset.get_numclasses()),
                          lstm_label)
    loss.backward()


    for p in lstm_model.parameters():
      p.data.add_(-lr, p.grad.data)
    total_loss += loss.data


    if batch_idx % args.log_interval == 0 and batch_idx > 0:
      cur_loss = total_loss[0] / args.log_interval
      elapsed = time.time() - start_time
      print('elapsed_time : {} | epoch {:3d} | {:6d}/{:6d} batches | lr {:02.6f} | ms/batch {:8.2f} | '
            'loss {:5.2f} | ppl {:8.2f}'.format(str(datetime.timedelta(seconds=time.time()-zero_batch_time)), epoch, batch_idx, len(train_data) // args.batch_size, lr,
                      elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
      total_loss = 0
      start_time = time.time()

def evaluate(loader):
  total_loss = 0
  hidden = lstm_model.init_hidden(args.batch_size)
  num_batchesprocessed = 0
  for batch_idx, (eval_seq, eval_label) in enumerate(loader):

    num_batchsamples = len(eval_seq[0]) # Hack, to avoid running into incomplete batches
    if num_batchsamples < args.batch_size:
      continue

    image_features = []
    lstm_label = Variable(torch.cat([eval_label.unsqueeze(0)] * args.sequence_length).view(-1).cuda())
    for image in eval_seq:
      image = Variable(image.cuda(), volatile=True)
      image_features.append(feature_cnn(image).squeeze().unsqueeze(0))

    lstm_input =  torch.cat(image_features) # seq_length x batch_size x feature_size
    lstm_input = Variable(lstm_input.data.cuda(), volatile=True)
    output, hidden = lstm_model(lstm_input, hidden)
    output_flat = output.view(-1, dataset.get_numclasses())
    total_loss += len(eval_seq) * lstm_criterion(output_flat, lstm_label).data
    hidden = repackage_hidden(hidden)
    num_batchesprocessed += 1
  return total_loss[0] / ((num_batchesprocessed)*args.batch_size)



# Loop over epochs.
lr = args.lr
prev_val_loss = None
for epoch in range(1, args.epochs+1):
    epoch_start_time = time.time()
    train()
    val_loss = evaluate(valdata_loader)
    print 'Finised Eval'
    sys.exit(0)
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
    prev_val_loss = val_loss
