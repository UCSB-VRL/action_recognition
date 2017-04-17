import os, sys
import torch
from torch.utils.data import Dataset

import imageio as io
import cv2
from sklearn.model_selection import StratifiedKFold
import numpy as np
from numpy.lib.stride_tricks import as_strided

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

def imread(p):
    img = io.imread(p)
    # opencv faster
    img = cv2.resize(img, (224, 224),
            interpolation=cv2.INTER_CUBIC)
    #img = imresize(img, (224, 224), 3)
    img = img.astype('float32')/255.0
    img -= mean
    img /= std
    return np.transpose(img, (2, 0, 1))

class ImageList(Dataset):
    def __init__(self, root, videos):
        self.root = root
        self.videos = videos

    def __getitem__(self, index):
        vid = self.videos[index] # path to video folder (of images)
        path = os.path.join(self.root, vid)
        img_list = os.listdir(path)
        img_list = [os.path.join(path, name)
                for name in sorted(img_list)]
        video = np.stack([imread(p) for p in img_list])
        return torch.from_numpy(video)

    def __len__(self):
        return len(self.videos)

class VideoList(Dataset):
    def __init__(self, root, videos, for_train=False, seq_length=16):
        self.root = root
        self.videos = videos
        self.for_train = for_train
        self.seq_length = seq_length

    # pick randomly 1 sequence per video to train,
    # pick evenly 20 sequences per video to validate/test
    def __getitem__(self, index):
        name, c = self.videos[index]
        path = os.path.join(self.root, name + '.npy')

        feat = np.load(path)
        n, d = feat.shape   # d=2048

        if self.for_train:
            start = np.random.randint(0, n-self.seq_length)
            feat = feat[start:start+self.seq_length]
            feat = feat[None, ...]            # RxLxD, R = 1
            frame_indexes = np.array(range(start, start+self.seq_length))
            frame_indexes = frame_indexes[None, ...]

        else:
            R = 20 # Sample the 20 sequences
            S = (n-self.seq_length) // (R-1)
            indexes = np.array(range(n))
            
            sn, sd = feat.strides
            i_sn, = indexes.strides

            feat = as_strided(feat, shape=(R, self.seq_length, d), strides=(S*sn, sn, sd))
            indexes = as_strided(indexes, shape=(R, self.seq_length), strides=(S*i_sn, i_sn))

            feat = np.ascontiguousarray(feat) # RxLxD, R = 20
            frame_indexes = np.ascontiguousarray(indexes) # RxL

        return feat, c, name, frame_indexes

    def __len__(self):
        return len(self.videos)


def collate(batch):
    x, y, z, w = zip(*batch)
    x = torch.cat([torch.from_numpy(a) for a in x])    # (bR)xLxD
    w = torch.cat([torch.from_numpy(a) for a in w])    # (bR)xL
    x = x.permute(1, 0, 2).contiguous()                # Lx(bR)xD
    y = torch.LongTensor(y)
    return x, y, z, w


def class_dict(ids_file):
    class2idx = {}
    with open(ids_file) as f:
        for line in f:
            c, name = line.split()
            class2idx[name] = int(c) - 1

    return class2idx


def video_list(data_file, class2idx):
    data = []
    with open(data_file) as f:
        for line in f:
            name = line.split()[0]
            name = os.path.splitext(name)[0]
            c = name.split('/')[0]
            c = class2idx[c]
            data.append((name, c))

    return data

def train_split(data, n_splits=5, select=0, seed=2017):
    labels = np.array([d[1] for d in data])
    rng = np.random.RandomState(seed)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
    cv = list(skf.split(labels, labels))
    train_index, valid_index = cv[select]
    train_data = [data[idx] for idx in train_index]
    valid_data = [data[idx] for idx in valid_index]
    return train_data, valid_data
