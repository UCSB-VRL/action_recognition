import os
import sys
import argparse
import numpy as np
import glob
from model import  cnn


parser = argparse.ArgumentParser(description='Compute feature mean for LSTM input from training data')
parser.add_argument('--feature-metadatafile', default='', dest='feature_metadatafile',
                    help='Metadata file which lists location of all feature files')
parser.add_argument('--feature-extractortype', action="store", dest='feature_extractortype',
                    default='vgg16', help="Type of feature extractor used on images")
parser.add_argument('--batch-size', action="store", dest='batch_size',
                    help="Batch size", default=32,
                    type=int)
parser.add_argument('--meanfile-location', default='', dest='meanfile_location',
                    help='Folder where the meanfile is stored')


args = parser.parse_args()

feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }

with open(args.feature_metadatafile, 'r') as f:
  seq_feature_files = f.read().splitlines()

feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype, batch_size=args.batch_size)
feature_dataloader = feature_extractor.create_dataloader(seq_feature_files, split_type='val', data_type='feature')

assert args.feature_extractortype in args.feature_metadatafile, 'Stored features dont correspond to \
                                                                                    feature_extractortype \
                                                                                    : %s'%args.feature_extractortype

feature_mean = np.zeros((1, feat_size_map[args.feature_extractortype]))

for batch_idx, (seq, label, _) in enumerate(feature_dataloader):

  if (batch_idx%10 == 0):
    print '\r %06d/%06d'%(batch_idx, len(seq_feature_files)//args.batch_size),
    sys.stdout.flush()

  array = seq.view(-1, feat_size_map[args.feature_extractortype]).mean(0).numpy()
  feature_mean = (batch_idx*feature_mean + array)/(batch_idx+1)

print ''
np.save(os.path.join(args.meanfile_location, 'train_mean.npy'), feature_mean)