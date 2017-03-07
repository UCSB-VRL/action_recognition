import os, sys, time
import numpy as np
from model import  cnn
from dataset import ucf101_dataset
import argparse
from torch.autograd import Variable


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
parser.add_argument('--nhid', type=int, default=200, dest='nhid'
                    help='humber of hidden units per layer in LSTM')

args = parser.parse_args()

feat_size_map = { 'vgg16': 4096
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

def train():
  for train_seq, train_label in traindata_loader:
      assert len(train_seq) == args.sequence_length
      for image in train_seq:
          image = Variable(image.cuda(), volatile=True)
          output = feature_cnn(image).squeeze()
      import ipdb; ipdb.set_trace()


      print '\r Feature extraction FPS : %6f'%(args.batch_size/(time.time() - t1)),
      sys.stdout.flush()
      t1 = time.time()