#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 15 16:41:27 2018

@author: xionghaipeng
"""

__author__ = 'xhp'

'''load the dataset'''
# from __future__ import print_function, division
import os
import torch
# import pandas as pd #load csv file
from skimage import io, transform  #
import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset  # , DataLoader#

# from torchvision import transforms, utils#
import glob  # use glob.glob to get special flielist
import scipy.io as sio  # use to import mat as dic,data is ndarray

import torch
import torch.nn.functional as F
import glob
import torchvision.transforms as trans
import cv2
from PIL import Image
from torch.utils.data import Dataset
import os
import numpy as np
import torch
from torchvision import transforms
import random
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

# Ignore warnings
import warnings

warnings.filterwarnings("ignore")


class myDataset(Dataset):
    """dataset. also can be used for annotation like density map"""

    def __init__(self, img_dir, tar_dir, rgb_dir, transform=None, if_test=False, \
                 IF_loadmem=False):
        """
        Args:
            img_dir (string ): Directory with all the images.
            tar_dir (string ): Path to the annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.IF_loadmem = IF_loadmem  # whether to load data in memory
        self.IF_loadFinished = False
        self.image_mem = []
        self.target_mem = []

        self.img_dir = img_dir
        self.tar_dir = tar_dir
        self.transform = transform

        mat = sio.loadmat(rgb_dir)
        self.rgb = mat['rgbMean'].reshape(1, 1, 3)  # rgbMean is computed after norm to [0-1]

        img_name = os.path.join(self.img_dir, '*.jpg')
        self.filelist = glob.glob(img_name)
        self.dataset_len = len(self.filelist)

        # for test process, load data is different
        self.if_test = if_test

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):

        # ------------------------------------
        # 1. see if load from disk or memory
        # ------------------------------------
        if (not self.IF_loadmem) or (not self.IF_loadFinished):
            img_name = self.filelist[idx]
            image = io.imread(img_name)  # load as numpy ndarray
            image = image / 255. - self.rgb  # to normalization,auto to change dtype

            (filepath, tempfilename) = os.path.split(img_name)
            (name, extension) = os.path.splitext(tempfilename)

            mat_dir = os.path.join(self.tar_dir, '%s.mat' % (name))
            mat = sio.loadmat(mat_dir)

            # if need to save in memory
            if self.IF_loadmem:
                self.image_mem.append(image)
                self.target_mem.append(mat)
                # updata if load finished
                if len(self.image_mem) == self.dataset_len:
                    self.IF_loadFinished = True

        else:
            image = self.image_mem[idx]
            mat = self.target_mem[idx]
            # target = mat['target']

        # for train may need pre load
        if not self.if_test:
            target = mat['crop_gtdens']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)

            # pad the image
            sample['image'], sample['target'] = get_pad(sample['image'], DIV=64), get_pad(sample['target'], DIV=64)
        else:
            target = mat['all_num']
            sample = {'image': image, 'target': target}
            if self.transform:
                sample = self.transform(sample)
            sample['density_map'] = torch.from_numpy(mat['density_map'])

            # pad the image
            sample['image'], sample['density_map'] = get_pad(sample['image'], DIV=64), get_pad(sample['density_map'],
                                                                                               DIV=64)

        return sample


######################################################################
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, target = sample['image'], sample['target']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'target': torch.from_numpy(target)}


######################################################################
def get_pad(inputs, DIV=64):
    h, w = inputs.size()[-2:]
    ph, pw = (DIV - h % DIV), (DIV - w % DIV)
    # print(ph,pw)

    if (ph != DIV) or (pw != DIV):
        tmp_pad = [pw // 2, pw - pw // 2, ph // 2, ph - ph // 2]
        # print(tmp_pad)
        inputs = F.pad(inputs, tmp_pad)

    return inputs


class SHTA(Dataset):
    def __init__(self, img_root, gt_dmap_root=None, bs=1, scale=[1], crop=1, randomcrop=False, down=1, raw=False,
                 preload=False, phase='train',maintrans=None, imgtrans=None, gttrans=None):
        '''
        img_root: the root path of img.
        gt_dmap_root: the root path of ground-truth density-map.
        gt_downsample: default is 0, denote that the output of deep-model is the same size as input image.
        phase: train or test
        '''
        self.img_root = img_root
        self.gt_dmap_root = gt_dmap_root
        self.crop = crop
        self.raw = raw
        self.phase = phase
        self.preload = preload
        self.bs = bs
        self.img_files = glob.glob(img_root + '/*.jpg')
        self.samples = len(self.img_files)
        self.maintrans = maintrans
        self.imgtrans = imgtrans
        self.gttrans = gttrans
        self.down = down
        self.randomcrop = randomcrop
        self.scale_list = scale
        if self.preload:
            self.data = read_image_and_gt(self.img_files, preload=True)

    def __len__(self):
        return self.samples

    def __getitem__(self, index):

        imgs = []
        gts = []
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if self.preload:
            imgname = self.data[index][0]
            img = self.data[index][1]
            gt = self.data[index][2]

        else:
            imgname, img, gt = read_image_and_gt(self.img_files[index], preload=False, mean=mean, std=std)

        # if self.maintrans is not None:
        #     img, den = self.maintrans(img, gt)
        # if self.imgtrans is not None:
        #     img = self.imgtrans(img)
        # if self.gttrans is not None:
        #     gt = self.gttrans(gt)

        if len(self.scale_list) > 1 and self.phase == 'train':
            temp = random.randint(0, len(self.scale_list)-1)
            scale_para = self.scale_list[temp]
            if  scale_para!=1:
                img = cv2.resize(img, (scale_para * img.shape[1], scale_para * img.shape[0]))
                gt = cv2.resize(gt, (scale_para * gt.shape[1], scale_para * gt.shape[0])) / scale_para / scale_para


        if random.random()>0.75 and self.phase == 'train':
            img = img[:, ::-1].copy()  # 水平翻转
            gt = gt[:, ::-1].copy()  # 水平翻转

        rh, rw,_ = img.shape
        ch, cw = rh // 2, rw // 2

        img = img.transpose([2, 0, 1])
        img = img[np.newaxis,:,:,:]
        gt = gt[np.newaxis,np.newaxis, :, :]
        gt = torch.tensor(gt, dtype=torch.float32)
        img = torch.tensor(img, dtype=torch.float32)

        if self.crop > 1 and self.randomcrop and self.phase == 'train':
            gts = []
            imgs = []
            count = 0
            while count < self.crop:
                sh = random.randint(0, ch)
                sw = random.randint(0, cw)
                img.append(imgs[:,:, sh:sh + ch, sw:sw + cw])
                gts.append(gts[:, :,sh:sh + ch, sw:sw + cw])
                count +=1

            imgs = [get_pad(i) for i in imgs]
            gts = [get_pad(i) for i in gts]
            img = torch.cat(imgs, 0)
            gt = torch.cat(gts, 0)

        elif self.crop > 1 and not self.randomcrop and self.phase == 'train':
            gts = []
            imgs = []

            imgs.append(img[:, :, 0:ch, 0:cw])
            gts.append(gt[:, :,0:ch, 0:cw])

            imgs.append(img[:, :,ch:, 0:cw])
            gts.append(gt[:, :, ch:, 0:cw])

            imgs.append(img[:, :, 0:ch, cw:])
            gts.append(gt[:, :, 0:ch, cw:])
            imgs.append(img[:, :, ch:, cw:])
            gts.append(gt[:, :,ch:, cw:])
            count = 4

            while count < self.crop:
                sh = random.randint(0, ch)
                sw = random.randint(0, cw)
                imgs.append(img[:, :, sh:sh + ch, sw:sw + cw])
                gts.append(gt[:, :, sh:sh + ch, sw:sw + cw])
                count +=1

            imgs = [get_pad(i) for i in imgs]
            gts = [get_pad(i) for i in gts]

            img = torch.cat(imgs, 0)
            gt = torch.cat(gts, 0)


        # img = get_pad(img)
        # gt = get_pad(gt)

        return imgname, img, gt


def read_image_and_gt(imgfiles, preload, mean, std):
    if preload:
        data = []
        print('Loading data into ram......')
        for idx, imgfile in tqdm(enumerate(imgfiles)):
            img = cv2.imread(imgfile)
            imgname = imgfile.split('/')[-1]
            img = img[:, :, ::-1].copy()
            img = img.astype(np.float32, copy=False)
            img = (img - mean) / std

            gtfile = imgfile.replace('original', 'knn_gt').replace('.jpg', '.npy').replace('images',
                                                                                           'ground-truth').replace(
                'IMG_',
                'GT_IMG_')
            gt = np.load(gtfile)

            # gt = torch.tensor(gt,dtype=torch.float32)
            # img = torch.tensor(img,dtype=torch.float32)
            #
            # img = get_pad(img)
            # gt = get_pad(gt)
            #
            # # rh, rw, c = img.shape
            # # dw = int(rw / 16) * 16
            # # dh = int(rh / 16) * 16
            # # img = cv2.resize(img, (dw, dh))
            # # zoom = rw * rh / dw / dh
            # # gt = cv2.resize(gt, (dw, dh), interpolation=cv2.INTER_CUBIC) * zoom
            #
            # img = img.transpose([2, 0, 1])
            # gt = torch.unsqueeze(gt,0)
            data.append([imgname, img, gt])

        return data

    else:
        img = cv2.imread(imgfiles)
        imgname = imgfiles.split('/')[-1]
        img = img[:, :, ::-1].copy()
        img = img.astype(np.float32, copy=False)
        img = (img / 255 - mean) / std
        gtfile = imgfiles.replace('original', 'knn_gt').replace('.jpg', '.npy').replace('images',
                                                                                        'ground-truth').replace('IMG_',
                                                                                                                'GT_IMG_')
        gt = np.load(gtfile)

        # gt = torch.tensor(gt, dtype=torch.float32)
        # img = torch.tensor(img, dtype=torch.float32)

        # img = get_pad(img)
        # gt = get_pad(gt)
        #
        # # rh, rw, c = img.shape
        # # dw = int(rw / 16) * 16
        # # dh = int(rh / 16) * 16
        # # img = cv2.resize(img, (dw, dh))
        # # zoom = rw * rh / dw / dh
        # # gt = cv2.resize(gt, (dw, dh), interpolation=cv2.INTER_CUBIC) * zoom
        #
        # img = img.transpose([2, 0, 1])
        # gt = torch.unsqueeze(gt,0)
        return imgname, img, gt


def load_data(img_path, train=True):
    gt_path = img_path.replace('.jpg', '.h5').replace('images', 'ground_truth')
    img = Image.open(img_path).convert('RGB')
    gt_file = h5py.File(gt_path)
    target = np.asarray(gt_file['density'])
    if False:
        crop_size = (img.size[0] / 2, img.size[1] / 2)
        if random.randint(0, 9) <= -1:

            dx = int(random.randint(0, 1) * img.size[0] * 1. / 2)
            dy = int(random.randint(0, 1) * img.size[1] * 1. / 2)
        else:
            dx = int(random.random() * img.size[0] * 1. / 2)
            dy = int(random.random() * img.size[1] * 1. / 2)

        img = img.crop((dx, dy, crop_size[0] + dx, crop_size[1] + dy))
        target = target[dy:crop_size[1] + dy, dx:crop_size[0] + dx]

        if random.random() > 0.8:
            target = np.fliplr(target)
            img = img.transpose(Image.FLIP_LEFT_RIGHT)

    target = cv2.resize(target, (target.shape[1] / 8, target.shape[0] / 8), interpolation=cv2.INTER_CUBIC) * 64

    return img, target


if __name__ == '__main__':
    inputs = torch.ones(6, 60, 730, 970);
    print('ori_input_size:', str(inputs.size()))
    inputs = get_pad(inputs);
    print('pad_input_size:', str(inputs.size()))

