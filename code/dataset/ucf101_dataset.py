import numpy as np
import os
import sys, time
import random
from glob import glob
import pandas as pd
import time


def _get_imglist(img_folder):
  images = glob(os.path.join(img_folder, '*.jpg'))
  return images

class ucf101Dataset:
  def __init__(self, data_root='', test_splitfile='', train_splitfile='', class_indfile='', seq_length=16, batch_size = 10):
    self._data_root = data_root
    self._test_splitfile = test_splitfile
    self._train_splitfile = train_splitfile
    self._seq_length = seq_length
    self._val_ratio = 0.1 # Number of validation samples picked from the training set
    self._batch_size = batch_size
    self._num_classes = 101

    # csv files which contain all the training, validation and test sequences possible
    self._train_seq_pkl = os.path.abspath('train_seq.csv')
    self._val_seq_pkl = os.path.abspath('val_seq.csv')
    self._test_seq_pkl = os.path.abspath('test_seq.csv')


    self._class_ind_dict = {}
    with open(class_indfile, 'r') as f:
      entries = f.read().splitlines()
    for e in entries:
      frags = e.split(' ')
      self._class_ind_dict[frags[1]] = int(frags[0])-1

  def get_numclasses(self):
    return self._num_classes

  def load_splits(self):
    """ Populate internal datastructures from the ucf101 splitfiles"""
    with open(self._test_splitfile, 'r') as f:
      videos = f.read().splitlines()
    self._test_videos_folders = [os.path.join(self._data_root, v.split('.')[0]) for v in videos]
    self._test_video_labels = [self._class_ind_dict[v.split('/')[0]] for v in videos]

    with open(self._train_splitfile, 'r') as f:
      videos = f.read().splitlines()
    random.seed(0)
    random.shuffle(videos)

    val_videos = videos[:int(self._val_ratio*len(videos))]
    self._val_videos_folders = [os.path.join(self._data_root, v.split('.')[0]) for v in val_videos]
    self._val_video_labels = [self._class_ind_dict[v.split('/')[0]] for v in val_videos]

    train_videos = videos[int(self._val_ratio*len(videos)):]
    self._train_videos_folders = [os.path.join(self._data_root, v.split('.')[0]) for v in train_videos]
    self._train_video_labels = [self._class_ind_dict[v.split('/')[0]] for v in train_videos]


  def create_seq_pkl(self, split_type='train'):
    assert split_type in  ['train', 'val', 'test']

    if split_type == 'train':
      pkl_file = self._train_seq_pkl
      video_folders = self._train_videos_folders
      video_labels = self._train_video_labels
    elif split_type == 'val':
      pkl_file = self._val_seq_pkl
      video_folders = self._val_videos_folders
      video_labels = self._val_video_labels
    else:
      pkl_file = self._test_seq_pkl
      video_folders = self._test_videos_folders
      video_labels = self._test_video_labels


    if os.path.exists(pkl_file):
      print 'Pickle file for subset %s with dataframe is already present at %s'%(split_type, pkl_file)
      return False

    print 'Preparing the csv file %s for subset : %s ....'%(pkl_file, split_type)
    seq_imgcolumns_lists = []
    for x in range(0, self._seq_length):
      seq_imgcolumns_lists.append([])

    df = pd.DataFrame()

    seq_label_lists = []
    for f_idx, f in enumerate(video_folders):
      if (f_idx%100 == 0):
        print '\r %s %06d/%06d'%(split_type, f_idx, len(video_folders)),
      images = _get_imglist(f)
      images.sort()
      if split_type == 'train':
        seq_startlocs = range(0, len(images)-self._seq_length, self._seq_length//4)
      else:
        seq_startlocs = range(0, len(images) - self._seq_length, self._seq_length)

      for loc in seq_startlocs:
        seq_images = images[loc:loc+self._seq_length]
        for i_idx, i in enumerate(seq_images):
          seq_imgcolumns_lists[i_idx].append(i)
        seq_label_lists.extend([video_labels[f_idx]])

    print ''
    df['label'] = seq_label_lists
    for x in range(0, self._seq_length):
      df['image_%03d'%x] = seq_imgcolumns_lists[x]
    df.to_csv(pkl_file, header=None, index=False)
    return True



  def load_sequence_data(self, split_type='train'):
    assert split_type in ['train', 'val', 'test']

    if split_type == 'train':
      pkl_file = self._train_seq_pkl
    elif split_type == 'val':
      pkl_file = self._val_seq_pkl
    else:
      pkl_file = self._test_seq_pkl

    print 'Loading sequence data for subset %s from %s'%(split_type, pkl_file),
    t1 = time.time()
    with open(pkl_file, 'r') as f:
      data = f.read().splitlines()
    print ' ... loaded in %6f seconds'%(time.time()-t1)
    return data



  
if __name__ == '__main__':
  ucf_dataset = ucf101Dataset(data_root='/media/brain/archith/video_analysis/UCF-101/UCF-101-images/',
                              test_splitfile='ucfTrainTestlist/testlist01.txt',
                              train_splitfile='ucfTrainTestlist/trainlist01.txt',
                              class_indfile='ucfTrainTestlist/classInd.txt')
  ucf_dataset.load_splits()
  ucf_dataset.create_seq_pkl(split_type='train')
  X = ucf_dataset.load_sequence_data(split_type='train')
