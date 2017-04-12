import numpy as np
import torch
import torch.utils.data as data
from torch import nn
from torch.autograd import Variable
import torchvision.models as models
import imageio as io
import os
import cv2
import sys, time
import pandas as pd
from skimage.transform import resize as imresize
from pedataloader import PEDataLoader

class featureExtractor:
  def __init__(self, cnn_name='vgg16'):
    self._cnn_name = cnn_name
  
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



  


