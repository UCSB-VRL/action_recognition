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


class SequenceList(data.Dataset):
    def __init__(self, data, root='', for_train=False):
        print('number of samples', len(data))

        self.root = root
        self.seqs = data
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]


    def __getitem__(self, index):
        frags = self.seqs[index].split(',')
        label = int(frags[0])
        paths = frags[1:]
        seq_images = []
        for p in paths: 
          image = io.imread(os.path.join(self.root, p))
          image = imresize(image, (224, 224), 3)
          image = image.astype('float32')/255.0
          image -= self.mean
          image /= self.std
          image = np.transpose(image, (2, 0, 1))

          seq_images.append(torch.from_numpy(image))
        
        return seq_images, label

    def __len__(self):
        return len(self.seqs)

class featureExtractor:
  def __init__(self, cnn_name='vgg16', batch_size=16):
    self._cnn_name = cnn_name
    self._b_size = batch_size
    self._loader_workers = max(batch_size // 2, 1)
  
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

  def create_dataloader(self, sequences, split_type='train'):
    assert split_type in  ['train', 'val', 'test']
    if split_type =='train':
      _for_train = True
      _shuffle = True
    else:
      _for_train = False
      _shuffle = False
    

    self._data_loader = torch.utils.data.DataLoader(
                          SequenceList(sequences,'', for_train=_for_train),
                          batch_size=self._b_size, shuffle=_shuffle,
                          num_workers=self._loader_workers, pin_memory=True)
    return self._data_loader


  


