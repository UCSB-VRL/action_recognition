import os, sys
import numpy as np
import argparse

import datetime
import time

import torch
from torch.autograd import Variable
from model import cnn
from dataset import data

parser = argparse.ArgumentParser(description='Extract features from dataset using desired CNN')
parser.add_argument('--dataset-root', action="store", dest='dataset_root',
                    help="Location where images extracted from the UCF-101 dataset are stored")
parser.add_argument('--feature-extractortype', action="store", dest='feature_extractortype',
                    default='resnet152', help="Type of feature extractor used on images")
parser.add_argument('--gpu-id', action="store", dest='gpu_id',
                    default='1', help="Index of GPU to be used")
parser.add_argument('--feature-root', default='', dest='feature_root',
                    help='Location where extracted features are stored')                    
parser.add_argument('--batch-size', action="store", dest='batch_size',
                    help="Batch size", default=32,
                    type=int)

args = parser.parse_args()
print('=' * 89)
print 'Command Line Arguments'
for x in args.__dict__.keys():
  print str(x) + ' : ' + str(args.__dict__[x])
print('=' * 89)

feature_extractor = cnn.featureExtractor(cnn_name=args.feature_extractortype)
feature_cnn = feature_extractor.create_cnn()

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

videos = []
classes = os.listdir(args.dataset_root)
classes.sort()
for c in classes:
    class_dir = os.path.join(args.dataset_root, c)
    vid_list = os.listdir(class_dir)
    vid_list.sort()
    videos.extend([os.path.join(c, v) for v in vid_list])

videodata_loader = torch.utils.data.DataLoader(data.ImageList(args.dataset_root, videos), 
                                               batch_size=1, num_workers=6,
                                               shuffle=False, pin_memory=True)

out_dir = args.feature_root
for c in classes:
    p = os.path.join(out_dir, c)
    if not os.path.exists(p):
        os.makedirs(p)

zero_batch_time = time.time()
# timers for FPS measurement
# do one video at a time
t1 = time.time()
frame_count = 0
for k, vid in enumerate(videodata_loader):
    vid = vid.squeeze(0) # 1xnx3x224x224 -> nx3x224x224
    vid = vid.cuda()
    out = []
    for x in vid.split(args.batch_size):
        n = x.size(0)
        x = Variable(x, volatile=True)
        y = feature_cnn(x).view(n, -1)
        out.append(y.data)
        frame_count += n


    out = torch.cat(out).cpu().numpy()
    name = videos[k]
    np.save(os.path.join(out_dir, name), out)
    if (k) % 10 == 0:
        elapsed_time =str(datetime.timedelta(seconds=time.time()-zero_batch_time))
        fps = frame_count/(time.time() - t1)
        t1 = time.time(); frame_count = 0
        print '\r Elapsed time : %s Features extracted from videos : %06d/%06d FPS: %.2f'%(elapsed_time, k, len(videos), fps),
        sys.stdout.flush()

