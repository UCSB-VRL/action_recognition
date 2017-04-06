import os, time, sys
import numpy as np
from model import  cnn
from model import lstm
from dataset import ucf101_dataset
import argparse
from torch.autograd import Variable
import torch.nn as nn
import torch
import datetime
import glob
import pandas as pd
from tensorflow.core.framework import summary_pb2
import tensorflow as tf
import collections

parser = argparse.ArgumentParser(description='Train or eval action recognition through LSTMs')
parser.add_argument('--dataset-root', action="store", dest='dataset_root',
                    help="Location where images extracted from the UCF-101 dataset are stored")
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
parser.add_argument('--operation-mode', default='feature', dest='operation_mode',
                    help='If "feature_mode" is set to "feature", then visual features are precomuted on dataset, \
                          if "feature_mode" is set to "image", then visual features are computed on the fly')
parser.add_argument('--feature-root', default='', dest='feature_root',
                    help='If "feature_mode" is set to "feature", then pre-computed features are saved under location  \
                          "feature_root"')
parser.add_argument('--momentum', default=0.9, type=float, dest='momentum',
                    help='momentum weight')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    dest='weight_decay', help='weight decay (default: 1e-4)')
parser.add_argument('--tfboard-dir',default='', type=str,
                    dest='tfboard_dir', help='Location where tensorboard plotting and trained model data is stored')
parser.add_argument('--lstm-dropout', default=0.0, type=float, dest='lstm_dropout',
                    help='LSTM dropout factor')
parser.add_argument('--lstm-init', default='uniform', type=str, dest='lstm_init',
                    help='type of init for LSTM decoder')
parser.add_argument('--run-name', default='', type=str, dest='run_name',
                    help='Run name')


args = parser.parse_args()
print('=' * 89)
print 'Command Line Arguments'
for x in args.__dict__.keys():
  print str(x) + ' : ' + str(args.__dict__[x])
print('=' * 89)


feat_size_map = { 'vgg16': 4096,
                  'resnet152': 2048
                }


dataset = ucf101_dataset.ucf101Dataset(data_root=args.dataset_root, 
                                       test_splitfile=args.test_splitfile,
                                       train_splitfile=args.train_splitfile, 
                                       class_indfile=args.class_indexfile,
                                       seq_length=args.sequence_length, 
                                       batch_size=args.batch_size)
dataset.load_splits()
dataset.create_seq_pkl(split_type='train')
dataset.create_seq_pkl(split_type='val')
dataset.create_seq_pkl(split_type='test')

feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype, batch_size=args.batch_size)
feature_cnn = feature_extractor.create_cnn()

# Fetch feature mean if available
featmean_present = False
feature_meanfile = os.path.join(args.feature_root, 'train_mean.npy')
if os.path.exists(feature_meanfile):
  featmean_present = True
  feature_mean = np.load(feature_meanfile)
  feature_mean = torch.Tensor(feature_mean).unsqueeze(0).repeat(args.sequence_length, 1, 1).cuda()
  print 'INFO: Loaded feature mean from : %s'%feature_meanfile
else:
   print 'INFO: No mean file found at %s'%feature_meanfile

def add_summary(tag, raw_value, global_step):
  value = summary_pb2.Summary.Value(tag=tag, simple_value=raw_value)
  summary = summary_pb2.Summary(value=[value])
  tf_summary_writer.add_summary(summary, global_step)

assert args.operation_mode in ['feature', 'image', 'image_seq']
print 'Mode of operation : %s'%args.operation_mode.upper()
if args.operation_mode == 'feature':
  # Precompute features from sequence images and store on disk for later processing as LSTM inputs
  split_names = ['val', 'train', 'test']
  split_nameidx_map = {}
  for idx, x in enumerate(split_names):
    split_nameidx_map[x] = idx

  seq_feature_listname_format = '%s_%s_seq_feature_filelist.csv'
  seq_df_filenames = []

  for split_idx, split_name in enumerate(split_names):
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
    splitseq_data = dataset.load_sequence_data(split_type=split_name)
    if (os.path.exists(seq_df_filename) and (len(curr_seq_featurefiles) == len(splitseq_data))):
      continue # features are already computed, associated metadata file also present

    # Get list of images within dataset and create a map from an image name to it's location in the list
    imgname_idx_dict = {}
    img_counter = 0
    split_imagelist = dataset.load_imglist_data(split_type=split_name)
    print 'Creating image_name to idx map ...',
    t1 = time.time()
    for s_i in split_imagelist:
      imgname_idx_dict[s_i.split(',')[1]] = img_counter
      img_counter += 1

    print 'done in %.3f seconds.'%(time.time()-t1)
    loader_batchsize = 32
    datasplit_features = np.zeros((len(split_imagelist), feat_size_map[args.feature_extractortype]))
    imagedata_loader = feature_extractor.create_dataloader(split_imagelist, split_type='val', data_type='image',
                                                           batch_size=loader_batchsize)
    processed_imgcount = 0
    zero_batch_time = time.time()
    print 'Extracting features from images for the split %s'%(split_name.upper())
    for batch_idx, (images, labels, image_paths) in enumerate(imagedata_loader):
      num_batch_features = len(image_paths)
      t1 = time.time()

      images = Variable(images.cuda(), volatile=True)
      seq_features = feature_cnn(images).squeeze().data.cpu().numpy()

      datasplit_features[processed_imgcount:processed_imgcount+num_batch_features,:] = seq_features
      processed_imgcount += num_batch_features
      elapse_time = time.time()-t1; t1 = time.time()
      if batch_idx%10 == 0:
        print '\r Elapsed time : %s Processing split : %s batch : %6d/%6d  | FPS: %.3f'%(
                                                              str(datetime.timedelta(seconds=
                                                                  time.time()-zero_batch_time)),
                                                              split_name, batch_idx,
                                                              len(split_imagelist)/loader_batchsize,
                                                              num_batch_features/elapse_time),
        sys.stdout.flush()

    print ''
    # Group all the features extracted from images into sequences according to 'seq_data'
    print 'Saving features on disk as sequence files for split %s ...'%(split_name)
    for seq_idx, seq in enumerate(splitseq_data):
      if (seq_idx %100 == 0):
          print '\r %06d/%06d'%(seq_idx, len(splitseq_data)),
          sys.stdout.flush()
      frags = seq.split(',')
      seq_label = frags[0]
      seq_images = frags[1:]
      seq_img_idxs = []
      for s_i in seq_images:
        seq_img_idxs.append(imgname_idx_dict[s_i])
      seq_features = datasplit_features[seq_img_idxs,:]
      seq_feature_file_name = seq_file_format%seq_idx
      np.save(seq_feature_file_name, seq_features)
      seq_feature_labels.append(seq_label)
      seq_feature_filenames.append(seq_feature_file_name)
    
    print ''
    seq_feature_df['label'] = seq_feature_labels
    seq_feature_df['sequence_feature_file'] = seq_feature_filenames
    seq_feature_df.to_csv( seq_df_filename, header=None, index=None)

  feature_dataloaders = []
  for idx, seq_df_f in enumerate(seq_df_filenames):
    split_type = seq_df_f.split('_')[0]
    with open(seq_df_f, 'r') as f:
      seq_feature_files = f.read().splitlines()
    feature_dataloaders.append(feature_extractor.create_dataloader(seq_feature_files, split_type=split_type, data_type='feature'))

  traindata_loader = feature_dataloaders[split_nameidx_map['train']]
  testdata_loader = feature_dataloaders[split_nameidx_map['test']]
  valdata_loader = feature_dataloaders[split_nameidx_map['val']]

  # Get the labels from the training set to get label-distribution
  train_labels = []
  with open(seq_df_filenames[split_nameidx_map['train']], 'r') as f:
    lines = f.readlines()
    for l in lines:
      train_labels.append(int(l.split(',')[0]))

elif args.operation_mode == 'image_seq':
  # Contains a list of strings of the format 'seq_label, seq_img1_path, seq_img2_path, seq_img3_path ...'
  train_data = dataset.load_sequence_data(split_type='train')
  test_data = dataset.load_sequence_data(split_type='test')
  val_data = dataset.load_sequence_data(split_type='val')
  traindata_loader = feature_extractor.create_dataloader(train_data, split_type='train', data_type='image_seq')
  testdata_loader = feature_extractor.create_dataloader(test_data, split_type='test', data_type='image_seq')
  valdata_loader = feature_extractor.create_dataloader(val_data, split_type='val', data_type='image_seq')

  # Get the labels from the training set to get label-distribution
  train_labels = []
  for l in train_data:
    train_labels.append(int(l.split(',')[0]))

else:
    assert 0, 'Not implemented'

#Figure out the relative frequencies of training data labels and inversely weigh the cost function
label_count_dict = dict(collections.Counter(train_labels))
labels_list = label_count_dict.keys(); labels_list.sort()
labels_hist = []
for l in labels_list:
  labels_hist.append(label_count_dict[l])
labels_importance = [sum(labels_hist)/float(x) for x in labels_hist]
labels_importance = np.array([x/sum(labels_importance) for x in labels_importance]) #normalize


lstm_model = lstm.RNNModel('LSTM', feat_size_map[args.feature_extractortype], args.nhid, args.nlayers,
                           dataset.get_numclasses(), args.lstm_dropout, args.lstm_init)
lstm_model.cuda()

lstm_criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(labels_importance).cuda())
lstm_optimizer = torch.optim.SGD(lstm_model.parameters(), args.lr,
                                 momentum=args.momentum,
                                 weight_decay=args.weight_decay)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def train():
  global global_step
  lstm_model.train()
  total_loss = 0; total_prec1 = 0; total_prec5 = 0
  start_time = time.time()
  zero_batch_time = time.time()

  for batch_idx, (train_seq, train_label, train_files) in enumerate(traindata_loader):
    num_batchsamples = train_seq.size(0)
    hidden = lstm_model.init_hidden(num_batchsamples)
    lstm_label = Variable(train_label.cuda())

    if args.operation_mode == 'image_seq':
      train_seq = Variable(train_seq.cuda(), volatile=True)
      train_seq = train_seq.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      seq_len, bz, num_channels, height, width = train_seq.size()
      train_seq = train_seq.view(bz*seq_len, num_channels, height, width)
      seq_features = feature_cnn(train_seq).squeeze()
      seq_features = seq_features.view(seq_len, bz, -1)
      lstm_input = Variable(seq_features.data.cuda(), volatile=False)

    elif args.operation_mode == 'feature':
      lstm_input = train_seq.float().squeeze()
      lstm_input = lstm_input.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      lstm_input = Variable(lstm_input.cuda(), volatile=False)
    else:
      assert 0, 'Not implemented'

    lstm_optimizer.zero_grad()
    # subtract training sequence mean is available
    if featmean_present:
      lstm_input.data = lstm_input.data - feature_mean.repeat(1, num_batchsamples, 1)

    output, hidden = lstm_model(lstm_input, hidden)
    
    # Consider only last time step's output for eval
    loss = lstm_criterion(output.narrow(0, args.sequence_length-1, 1).squeeze(),
                          lstm_label)
    #train_prec1, train_prec5 = accuracy(output.view(-1, dataset.get_numclasses()).data, lstm_label.data, topk=(1, 5))
    train_prec1, train_prec5 = accuracy(output.narrow(0, args.sequence_length-1, 1).squeeze().data, lstm_label.data, topk=(1, 5))
    total_prec1 += train_prec1[0]; total_prec5 += train_prec5[0]
    loss.backward()
    lstm_optimizer.step()
    global_step += 1
    total_loss += loss.data

    if batch_idx % args.log_interval == 0 and batch_idx > 0:
      cur_loss = total_loss[0] / args.log_interval
      curr_prec1 = total_prec1/ args.log_interval; curr_prec5 = total_prec5/ args.log_interval
      elapsed = time.time() - start_time
      print('elapsed_time : {} | epoch {:3d} | {:6d}/{:6d} batches | lr {:02.6f} | ms/batch {:8.2f} | '
            'loss {:5.2f} | top1 {:8.2f} | top5 {:8.2f}'.format(str(datetime.timedelta(seconds=time.time()-zero_batch_time)),
                                                                epoch,
                                                                batch_idx,
                                                                len(traindata_loader),
                                                                lr,
                                                                elapsed * 1000 / args.log_interval,
                                                                cur_loss,
                                                                curr_prec1,
                                                                curr_prec5))

      # Log data to tensorflow tensorboard
      add_summary(tag="Train_Prec1", raw_value=curr_prec1, global_step=global_step)
      add_summary(tag="Train_Prec5", raw_value=curr_prec5, global_step=global_step)
      add_summary(tag="Train_Loss", raw_value=cur_loss, global_step=global_step)
      add_summary(tag="Learning_Rate", raw_value=lr, global_step=global_step)

      total_loss = 0
      total_prec1 = 0
      total_prec5 = 0
      start_time = time.time()

def evaluate(loader):
  total_loss = 0
  lstm_model.eval()
  num_batchesprocessed = 0
  total_prec1 = 0; total_prec5 = 0
  for batch_idx, (eval_seq, eval_label, eval_files) in enumerate(loader):  
    num_batchsamples = eval_seq.size(0)
    hidden = lstm_model.init_hidden(num_batchsamples)
    lstm_label = Variable(eval_label.cuda())

    if args.operation_mode == 'image_seq':
      eval_seq = Variable(eval_seq.cuda(), volatile=True)
      eval_seq = eval_seq.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      seq_len, bz, num_channels, height, width = eval_seq.size()
      eval_seq = eval_seq.view(bz*seq_len, num_channels, height, width)
      seq_features = feature_cnn(eval_seq).squeeze()
      seq_features = seq_features.view(seq_len, bz, -1)
      lstm_input = Variable(seq_features.data.cuda(), volatile=False)

    elif args.operation_mode == 'feature':
      lstm_input = eval_seq.float().squeeze()
      lstm_input = lstm_input.transpose(0, 1).contiguous()  # swap batch_size and seq_len dimensions
      lstm_input = Variable(lstm_input.cuda())

    else:
        assert 0, 'Not implemented'

    # subtract training sequence mean is available
    if featmean_present:
      lstm_input.data = lstm_input.data - feature_mean.repeat(1, num_batchsamples, 1)
    
    output, hidden = lstm_model(lstm_input, hidden)
    #output_flat = output.view(-1, dataset.get_numclasses())
    output_flat = output.narrow(0, args.sequence_length-1, 1).squeeze()
    eval_prec1, eval_prec5 = accuracy(output_flat.data, lstm_label.data, topk=(1, 5))
    total_prec1 += eval_prec1[0]; total_prec5 += eval_prec5[0]

    total_loss += len(eval_seq) * lstm_criterion(output_flat, lstm_label).data
    num_batchesprocessed += 1

  return total_loss[0] / ((num_batchesprocessed)*args.batch_size), total_prec1/num_batchesprocessed, total_prec5/num_batchesprocessed

def adjust_learning_rate(optimizer, _lr):
  """Sets the learning rate to the initial LR decayed by 10"""
  _lr = _lr*0.5
  for param_group in optimizer.param_groups:
    param_group['lr'] = _lr
  return _lr

def adjust_learning_rate2(optimizer, _epoch):
  """Sets the learning rate to the initial LR decayed by 10"""
  _lr = args.lr * 0.1 ** (_epoch // 15)
  for param_group in optimizer.param_groups:
    param_group['lr'] = _lr
  return _lr

# tensorboard structures
if args.run_name == '':
  foldername = str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')
else:
  foldername = args.run_name + '_' + str(datetime.datetime.now()).replace(' ', '_').replace(':', '_')

tf_summary_folder = os.path.join(args.tfboard_dir, foldername)
tf_summary_writer = tf.summary.FileWriter(tf_summary_folder)
model_save_path = os.path.join(tf_summary_folder, 'lstm_model_%06d.pt')
global_step = 0

# Loop over epochs.
lr = args.lr
best_val_loss = None
saved_epoch = 0
# At any point you can hit Ctrl + C to break out of training early.
try:
  prev_val_loss = None
  for epoch in range(1, args.epochs+1):
      #lr = adjust_learning_rate2(lstm_optimizer, epoch)
      epoch_start_time = time.time()
      train()      
      val_loss, val_top1, val_top5 = evaluate(valdata_loader)

      # Log data to tensorflow tensorboard
      add_summary(tag="Val_Prec1", raw_value=val_top1, global_step=global_step)
      add_summary(tag="Val_Prec5", raw_value=val_top5, global_step=global_step)
      add_summary(tag="Val_Loss", raw_value=val_loss, global_step=global_step)

      print 'Finised Eval'
      print('-' * 89)
      print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'top1 {:8.2f} | top5 {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, val_top1, val_top5))
      print('-' * 89)
      # Save the model if the validation loss is the best we've seen so far.
      if not best_val_loss or val_loss < best_val_loss:
        saved_epoch = epoch
        with open(model_save_path%saved_epoch, 'wb') as f:
          torch.save(lstm_model, f)
        best_val_loss = val_loss
      # Anneal the learning rate if no improvement has been seen in the validation dataset.
      else:
        lr = adjust_learning_rate(lstm_optimizer, lr)
        pass

      if lr < 1e-6:
        print('Exiting from training due to low value of learning rate')
        break

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
print('Loading model from %s for test'%model_save_path%saved_epoch)
with open(model_save_path%saved_epoch, 'rb') as f:
  lstm_model = torch.load(f)

# Run on test data.
test_loss, test_top1, test_top5  = evaluate(testdata_loader)

add_summary(tag="Test_Prec1", raw_value=test_top1, global_step=global_step)
add_summary(tag="Test_Prec5", raw_value=test_top5, global_step=global_step)
add_summary(tag="Test_Loss", raw_value=test_loss, global_step=global_step)

print('=' * 89)
print('| End of training | test loss {:5.2f} | top1 {:8.2f} | top5 {:8.2f}'.format(
    test_loss, test_top1, test_top5))
print('=' * 89)
