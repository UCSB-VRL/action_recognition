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
import pickle
import glob
import pandas as pd

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
parser.add_argument('--operation_mode', default='feature', dest='operation_mode',
                    help='If "feature_mode" is set to "feature", then visual features are precomuted on dataset, \
                          if "feature_mode" is set to "image", then visual features are computed on the fly')
parser.add_argument('--feature_root', default='', dest='feature_root',
                    help='If "feature_mode" is set to "feature", then pre-computed features are saved under location  \
                          "feature_root"')

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

# Contains a list of strings of the format 'seq_label, seq_img1_path, seq_img2_path, seq_img3_path ...'
train_data = dataset.load_sequence_data(split_type='train')
test_data = dataset.load_sequence_data(split_type='test')
val_data = dataset.load_sequence_data(split_type='val')

feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype, batch_size=args.batch_size)
feature_cnn = feature_extractor.create_cnn()



assert args.operation_mode in ['feature', 'image']
print 'Mode of operation : %s'%args.operation_mode.upper()
if args.operation_mode == 'feature':
  # Precompute features from sequence images and store on disk for later processing as LSTM inputs
  curr_seq_featurefiles = glob.glob(os.path.join(args.feature_root, '*.npy'))
  split_names = ['train', 'test', 'val']
  split_nameidx_map = {}
  for idx, x in enumerate(split_names):
    split_nameidx_map[x] = idx

  seq_feature_listname_format = '%s_%s_seq_feature_filelist.csv'
  seq_df_filenames = []

  for split_idx, data in enumerate([train_data, test_data, val_data]):
    seq_feature_df = pd.DataFrame(columns=['label', 'sequence_feature_file'])
    seq_feature_labels = []
    seq_feature_filenames = []
    seq_feature_counter = 0
    seq_file_folder = os.path.join(args.feature_root, split_names[split_idx])
    if not os.path.exists(seq_file_folder):
      os.makedirs(seq_file_folder)

    seq_file_format = os.path.join(seq_file_folder, '%07d.npy')
    curr_seq_featurefiles = glob.glob(os.path.join(seq_file_folder, '*.npy'))
    seq_df_filename = seq_feature_listname_format%(split_names[split_idx], args.feature_extractortype)
    seq_df_filenames.append(seq_df_filename)
    if (os.path.exists(seq_df_filename) and (len(curr_seq_featurefiles) == len(data))):
      continue # features are already computed, associated metadata file also present

    imagedata_loader = feature_extractor.create_dataloader(data, split_type='val', data_type='image')
    zero_batch_time = time.time()
    for batch_idx, (seq, label, folder) in enumerate(imagedata_loader):
      t1 = time.time()
      seq = Variable(seq.cuda(), volatile=True)
      seq = seq.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      seq_len, bz, num_channels, height, width = seq.size()
      seq = seq.view(bz*seq_len, num_channels, height, width)
      seq_features = feature_cnn(seq).squeeze()
      seq_features = seq_features.view(seq_len, bz, -1)

      for b_index in range(bz):
        seq_feature_filename = os.path.join(seq_file_folder, seq_file_format%seq_feature_counter)
        np.save(seq_feature_filename, seq_features.narrow(1, b_index, 1).squeeze().data.cpu().numpy())
        seq_feature_filenames.append(seq_feature_filename)
        seq_feature_labels.append(label[b_index])
        seq_feature_counter += 1

      elapse_time = time.time()-t1; t1 = time.time()
      print 'Elapsed time : %s Processing split : %s batch : %6d/%6d  | FPS: %.3f'%(
                                                            str(datetime.timedelta(seconds=
                                                                time.time()-zero_batch_time)),
                                                                split_names[split_idx], batch_idx,
                                                                  len(data)/args.batch_size,
                                                                  bz*args.sequence_length/elapse_time)
    seq_feature_df['label'] = seq_feature_labels
    seq_feature_df['sequence_feature_file'] = seq_feature_filenames
    seq_feature_df.to_csv( seq_df_filename, header=None, index=None)

  feature_dataloaders = []
  for idx, seq_df_f in enumerate(seq_df_filenames):
    split_type = seq_df_f.split('_')[0]
    with open(seq_df_f, 'r') as f:
      seq_feature_files = f.read().splitlines()
    feature_dataloaders.append(feature_extractor.create_dataloader(seq_feature_files, split_type=split_type, data_type='features'))

  traindata_loader = feature_dataloaders[split_nameidx_map['train']]
  testdata_loader = feature_dataloaders[split_nameidx_map['test']]
  valdata_loader = feature_dataloaders[split_nameidx_map['test']]

else:
  traindata_loader = feature_extractor.create_dataloader(train_data, split_type='train', data_type='image')
  testdata_loader = feature_extractor.create_dataloader(test_data, split_type='test', data_type='image')
  valdata_loader = feature_extractor.create_dataloader(val_data, split_type='val', data_type='image')


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

  for batch_idx, (train_seq, train_label, _) in enumerate(traindata_loader):
    num_batchsamples = train_seq.size(0)  # Hack, to avoid running into incomplete batches
    if num_batchsamples < args.batch_size:
      continue

    lstm_label = Variable(torch.cat([train_label.unsqueeze(0)] * args.sequence_length).view(-1).cuda())

    if args.operation_mode == 'image':
      train_seq = Variable(train_seq.cuda(), volatile=True)
      train_seq = train_seq.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      seq_len, bz, num_channels, height, width = train_seq.size()
      train_seq = train_seq.view(bz*seq_len, num_channels, height, width)
      seq_features = feature_cnn(train_seq).squeeze()
      seq_features = seq_features.view(seq_len, bz, -1)
      lstm_input = Variable(seq_features.data.cuda(), volatile=False)

    else:
      lstm_input = train_seq.squeeze()
      lstm_input = lstm_input.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      lstm_input = Variable(lstm_input.cuda(), volatile=False)

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
  for batch_idx, (eval_seq, eval_label, _) in enumerate(loader):
    num_batchsamples = eval_seq.size(0)  # Hack, to avoid running into incomplete batches
    if num_batchsamples < args.batch_size:
      continue

    lstm_label = Variable(torch.cat([eval_label.unsqueeze(0)] * args.sequence_length).view(-1).cuda())

    if args.operation_mode == 'image':
      eval_seq = Variable(eval_seq.cuda(), volatile=True)
      eval_seq = eval_seq.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      seq_len, bz, num_channels, height, width = eval_seq.size()
      eval_seq = eval_seq.view(bz*seq_len, num_channels, height, width)
      seq_features = feature_cnn(eval_seq).squeeze()
      seq_features = seq_features.view(seq_len, bz, -1)
      lstm_input = Variable(seq_features.data.cuda(), volatile=False)

    else:
      lstm_input = eval_seq.squeeze()
      lstm_input = lstm_input.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      lstm_input = Variable(lstm_input.cuda())

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
    print('-' * 89)
    print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
            'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                       val_loss, math.exp(val_loss)))
    print('-' * 89)
    # Anneal the learning rate.
    if prev_val_loss and val_loss > prev_val_loss:
        lr /= 4
    prev_val_loss = val_loss
