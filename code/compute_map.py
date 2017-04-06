""" Script to compute mean average precision for UCF101 test data results"""
import os, time
import numpy as np
from model import  cnn
from model import lstm
from dataset import ucf101_dataset
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch
import sys
import glob
from sklearn.metrics import average_precision_score
from sklearn.metrics import confusion_matrix
import datetime
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compute Mean average precision scores for the UCF-101 dataset')
parser.add_argument('--dataset-root', action="store", dest='dataset_root',
                    help="Location where images extracted from the UCF-101 dataset are stored")
parser.add_argument('--test_splitfile', action="store", dest='test_splitfile',
                    help="Location the test split of dataset is specified")
parser.add_argument('--train-splitfile', action="store", dest='train_splitfile',
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
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--log-interval', type=int, default=20, dest='log_interval',
                    help='report interval')
parser.add_argument('--feature_root', default='', dest='feature_root',
                    help='If "operation_mode" is set to "feature", then pre-computed features are saved under location  \
                          "feature_root"')
parser.add_argument('--load-model', type=str,  dest='load_model', default='lstm_saved_model.pt',
                    help='path to load the model to be tested')
parser.add_argument('--class-indexfile', type=str,  dest='class_indexfile', default='',
                    help='File where class indices and class names are stored')
parser.add_argument('--dataset-subset', type=str,  dest='dataset_subset', default='test',
                    help='Subset of the dataset that needs to be tested calc')

args = parser.parse_args()
print('=' * 89)
print 'Command Line Arguments'
for x in args.__dict__.keys():
  print str(x) + ' : ' + str(args.__dict__[x])
print('=' * 89)

feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }

assert args.dataset_subset in ['train', 'test', 'val']

#Load class name and indices
with open(args.class_indexfile, 'r') as f:
  class_idx_name = f.read().splitlines()

# Load the best saved model.
with open(args.load_model, 'rb') as f:
  lstm_model = torch.load(f)

# Fetch feature mean if available
featmean_present = False
feature_meanfile = os.path.join(args.feature_root, 'train_mean.npy')
if os.path.exists(feature_meanfile):
  featmean_present = True
  feature_mean = np.load(feature_meanfile)
  feature_mean = torch.Tensor(feature_mean).cuda()
  print 'INFO: Loaded feature mean from : %s'%feature_meanfile
else:
   print 'INFO: No mean file found at %s'%feature_meanfile
   

test_dataset = ucf101_dataset.ucf101Dataset(data_root=args.dataset_root, 
                                            test_splitfile=args.test_splitfile,
                                            train_splitfile=args.train_splitfile, 
                                            class_indfile=args.class_indexfile,
                                            seq_length=args.sequence_length, 
                                            batch_size=args.batch_size)
test_dataset.load_splits()
video_folders, video_labels = test_dataset.get_videodata(args.dataset_subset)

dataset_vid_imgs = []
imgname_idx_dict = {}
img_counter = 0
if not os.path.exists(args.feature_root):
  print 'Folder for feature file : %s does not exist!'%args.feature_root
  sys.exit(0)
dataset_features_file = os.path.join(args.feature_root, '%sset_features.npy'%args.dataset_subset)

print 'Creating image_name to idx map ...',
t1 = time.time()
for vid_idx, vid_name in enumerate(video_folders):
  images = glob.glob(os.path.join(vid_name, '*.jpg'))
  images.sort()
  for i in images:
      imgname_idx_dict[i] = img_counter
      img_counter += 1
      dataset_vid_imgs.append('0,' + i)

print 'in %4.3f seconds ...'%(time.time()-t1)

def extract_features(loader, _feature_cnn):
  start_time = time.time()
  _dataset_features = np.zeros((len(dataset_vid_imgs), feat_size_map[args.feature_extractortype]))
  count = 0
  t0 = time.time()
  for batch_idx, (eval_batch, eval_label, img_paths) in enumerate(loader):
    num_batchsamples = len(img_paths)

    if (batch_idx% 10 == 0):
      elapsed_time = time.time() - start_time
      print '\r elapsed_time : %s |  Processing batch %05d/%05d FPS: %3f'% (str(datetime.timedelta(seconds=time.time()-t0)),
                                                                            batch_idx,
                                                                            len(dataset_vid_imgs)//args.batch_size,
                                                                            10*args.batch_size/(elapsed_time)),
      start_time = time.time()
      sys.stdout.flush()

    eval_batch = Variable(eval_batch.cuda(), volatile=True)
    batch_features = _feature_cnn(eval_batch).squeeze()
    # subtract training sequence mean is available
    if featmean_present:
      batch_features.data = batch_features.data - feature_mean.repeat(num_batchsamples, 1)

    _dataset_features[count:count+num_batchsamples, : ] = batch_features.data.cpu().numpy()
    count += num_batchsamples

  return _dataset_features

if not os.path.exists(dataset_features_file):
  feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype, batch_size=args.batch_size)
  feature_cnn = feature_extractor.create_cnn()
  # Split type is set to 'test' to ensure that files are processed in order, irrespective of actual data split
  test_dataloader = feature_extractor.create_dataloader(dataset_vid_imgs, split_type='test', data_type='image')
  dataset_features = extract_features(test_dataloader, feature_cnn)
  np.save(dataset_features_file, dataset_features)
else:
  print 'Loading dataset features file : %s ....'%dataset_features_file,
  sys.stdout.flush()
  t1 = time.time()
  dataset_features = np.load(dataset_features_file)
  print 'in %3.3f seconds'%(time.time()-t1)

print 'Dataset image feature computation done!'

gt_data = np.zeros((len(video_folders), test_dataset.get_numclasses()))
result_data = np.zeros((len(video_folders), test_dataset.get_numclasses()))

lstm_model.eval()
softmax = torch.nn.Softmax()

t0 = time.time()
# Extract the features from the pre-computed from data
for vid_idx, vid_name in enumerate(video_folders):
  images = glob.glob(os.path.join(vid_name, '*.jpg'))
  images.sort()
  seq_start_locs = range(0, len(images)-args.sequence_length, args.sequence_length//2)
  if (vid_idx%10 == 0):
    print '\r elapsed_time : %s |  Video : %05d/%05d'%(str(datetime.timedelta(seconds=time.time()-t0)),
                                                      vid_idx, len(video_folders)),
    sys.stdout.flush()

  seq_features = np.array([])
  for l_idx, l in enumerate(seq_start_locs):
    img_feature_idxs = []
    for im in images[l:l+args.sequence_length]:
      img_feature_idxs.append(imgname_idx_dict[im])


    if seq_features.size == 0:
      seq_features = np.expand_dims(dataset_features[img_feature_idxs, :], 0)
    else:
      seq_features = np.vstack((seq_features, np.expand_dims(dataset_features[img_feature_idxs, :], 0)))

  num_batches = int(np.ceil(float(seq_features.shape[0])/args.batch_size))
  vid_seq_results = np.zeros((num_batches, test_dataset.get_numclasses()))

  for b_idx in range(0, num_batches):
    b_start = b_idx*args.batch_size
    b_end = min((b_idx+1)*args.batch_size, seq_features.shape[0])
    valid_seqs = b_end-b_start
    b_seq_features = seq_features[b_start:b_end, :]

    hidden = lstm_model.init_hidden(valid_seqs)
    lstm_input = Variable(torch.from_numpy(b_seq_features).float(), volatile=False)
    lstm_input = lstm_input.transpose(0,1).contiguous().cuda()
    lstm_output, hidden = lstm_model(lstm_input, hidden)
    softmax_input = lstm_output.narrow(0, args.sequence_length-1, 1).squeeze()
    if len(tuple(softmax_input.size())) == 1:
      # Softmax call requires 2D tensor input
      softmax_input = softmax_input.unsqueeze(0)

    class_scores = softmax(softmax_input).data
    # Mean along frame axis in a sequence
    class_scores = class_scores.mean(0).cpu().numpy()

    vid_seq_results[b_idx, :] = class_scores

  video_score = np.mean(vid_seq_results, axis=0)
  result_data[vid_idx, :] = video_score
  gt_data[vid_idx, video_labels[vid_idx]] = 1.0

print ''
average_precision = [0.0]*test_dataset.get_numclasses()
for class_idx in range(0, test_dataset.get_numclasses()):
  #average_precision[class_idx] = average_precision_score(gt_data[:, class_idx], result_data[:, class_idx])
  average_precision[class_idx] = average_precision_score(gt_data[:, class_idx], result_data[:, class_idx])
  print 'Class %3s | %25s | Average Precision : %.3f'%(class_idx_name[class_idx].split(' ')[0],
                                                       class_idx_name[class_idx].split(' ')[1],
                                                       average_precision[class_idx])

print ''
print 'Mean Average Precision : %.3f'%(np.mean(average_precision))

print 'Computing confusion matrix'
gt_labels = np.argmax(gt_data, axis=1)
result_labels = np.argmax(result_data, axis=1)
conf_matrix = confusion_matrix(gt_labels, result_labels)
norm_conf = np.zeros(conf_matrix.shape)
for r_idx, row in enumerate(conf_matrix):
  norm_conf[r_idx, :] = conf_matrix[r_idx,:]/float(sum(conf_matrix[r_idx,:]))

fig = plt.figure()
plt.clf()
ax = fig.add_subplot(111)
ax.set_aspect(1)
res = ax.imshow(np.array(norm_conf), cmap=plt.cm.jet, 
                interpolation='nearest')
width, height = conf_matrix.shape

import ipdb; ipdb.set_trace()
cb = fig.colorbar(res)
print 'Saving confusion matrix as an image'
plt.savefig('confusion_matrix.png', format='png')













