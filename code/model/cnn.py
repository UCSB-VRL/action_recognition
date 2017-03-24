import numpy as np
import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
import imageio as io
import os
import sys, time
import pandas as pd
from skimage.transform import resize as imresize
from pedataloader import PEDataLoader


class ImageSequenceList(data.Dataset):
    def __init__(self, data, root='', for_train=False):
        print('number of samples', len(data))
        self.root = root
        self.seqs = data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, index):
        frags = self.seqs[index].split(',')
        paths = frags[1:]
        label = int(frags[0])
        seq_images = []
        folder_names = []
        for p in paths:
          folder_names.append(os.path.dirname(p))
          image = io.imread(os.path.join(self.root, p))
          image = imresize(image, (224, 224), 3)
          image = image.astype('float32')/255.0
          image -= self.mean
          image /= self.std
          image = np.transpose(image, (2, 0, 1))

          seq_images.append(torch.from_numpy(image))

        seq_images = torch.stack(seq_images)
        assert len(set(folder_names)) == 1, 'All images in a sequence need to be from the same folder!'
        return seq_images, label, folder_names[0]
    
    def __call__(self, index):
      return self.__getitem__(index)

    def __len__(self):
        return len(self.seqs)


class ImageList(data.Dataset):
    def __init__(self, data, root='', for_train=False):
        print('number of samples', len(data))
        self.root = root
        self.imgs = data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

    def __getitem__(self, index):
        frags = self.imgs[index].split(',')
        path = frags[1]
        label = int(frags[0])

        image = io.imread(os.path.join(self.root, path))
        image = imresize(image, (224, 224), 3)
        image = image.astype('float32')/255.0
        image -= self.mean
        image /= self.std
        image = np.transpose(image, (2, 0, 1))

        return torch.from_numpy(image), label, path

    def __call__(self, index):
      return self.__getitem__(index)

    def __len__(self):
        return len(self.imgs)




class FeatureSequenceList(data.Dataset):
  def __init__(self, data, root='', for_train=False):
    print('number of samples', len(data))
    self.root = root
    self.seq_files = data

  def __getitem__(self, index):
    frags = self.seq_files[index].split(',')
    # Expected line format : 'label_number, numpy_file_path'
    assert(len(frags) == 2)
    label = int(frags[0])
    numpy_fpath = frags[1]
    numpy_arr = np.load(numpy_fpath)
    seq_data = torch.from_numpy(numpy_arr)

    return seq_data, label, numpy_fpath

  def __call__(self, index):
    return self.__getitem__(index)

  def __len__(self):
    return len(self.seq_files)


class featureExtractor:
  def __init__(self, cnn_name='vgg16', batch_size=16):
    self._cnn_name = cnn_name
    self._b_size = batch_size
    self._loader_workers = batch_size
  
  def create_cnn(self):
    print 'Creating CNN of type  %s '%(self._cnn_name),
    t1 = time.time()
    if self._cnn_name == 'vgg16':
      vgg16 = getattr(models, 'vgg16')(pretrained=True)
      self._cnn = vgg16
      self._cnn.classifier = nn.Sequential(
        *list(vgg16.classifier.children())[:-1]) #to get hold of fc7
    elif self._cnn_name == 'resnet152':
      resnet = getattr(models, 'resnet152')(pretrained=True)
      self._cnn = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, 
                                resnet.maxpool, resnet.layer1, resnet.layer2, 
                                resnet.layer3, resnet.layer4, resnet.avgpool)      
    else:
      print 'CNN architecture of type : %s not currently implemented'%self._cnn_name
      assert 0
    print ' ... created in %6f seconds'%(time.time()-t1)
    self._cnn.eval()
    self._cnn.cuda()



    return self._cnn

  def create_dataloader(self, sequences, split_type='train', data_type='image', batch_size=None):
    assert split_type in  ['train', 'val', 'test']
    if split_type =='train':
      _for_train = True
      _shuffle = True
    else:
      _for_train = False
      _shuffle = False

    if isinstance(batch_size, int):
      bsize = batch_size
      load_workers = bsize
    else:
      bsize = self._b_size
      load_workers = self._loader_workers

    assert data_type in ['image_seq', 'feature', 'image']

    if data_type == 'image_seq':
      _data_loader = torch.utils.data.DataLoader(ImageSequenceList(sequences,'', for_train=_for_train),
                                                 batch_size=bsize, shuffle=_shuffle,
                                                 num_workers=load_workers, pin_memory=True)

    elif data_type == 'feature':
      _data_loader = torch.utils.data.DataLoader(FeatureSequenceList(sequences,'', for_train=_for_train),
                                                 batch_size=bsize, shuffle=_shuffle,
                                                 num_workers=load_workers, pin_memory=True)

    else:
      _data_loader = torch.utils.data.DataLoader(ImageList(sequences,'', for_train=_for_train),
                                                 batch_size=bsize, shuffle=_shuffle,
                                                 num_workers=load_workers, pin_memory=True)

    return _data_loader



  


