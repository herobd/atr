import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math

from . import grid_distortion

from utils import string_utils, safe_load, augmentation

import random
PADDING_CONSTANT = 0

def collate(batch):
    batch = [b for b in batch if b is not None]
    #These all should be the same size or error
    assert len(set([b['line_img'].shape[0] for b in batch])) == 1
    assert len(set([b['line_img'].shape[2] for b in batch])) == 1

    dim0 = batch[0]['line_img'].shape[0]
    dim1 = max([b['line_img'].shape[1] for b in batch])
    dim2 = batch[0]['line_img'].shape[2]

    all_labels = []
    label_lengths = []

    input_batch = np.full((len(batch), dim0, dim1, dim2), PADDING_CONSTANT).astype(np.float32)
    for i in range(len(batch)):
        b_img = batch[i]['line_img']
        input_batch[i,:,:b_img.shape[1],:] = b_img

        l = batch[i]['gt_label']
        all_labels.append(l)
        label_lengths.append(len(l))

    all_labels = np.concatenate(all_labels)
    label_lengths = np.array(label_lengths)

    line_imgs = input_batch.transpose([0,3,1,2])
    line_imgs = torch.from_numpy(line_imgs)
    labels = torch.from_numpy(all_labels.astype(np.int32))
    label_lengths = torch.from_numpy(label_lengths.astype(np.int32))

    return {
        "line_imgs": line_imgs,
        "labels": labels,
        "label_lengths": label_lengths,
        "gt": [b['gt'] for b in batch]
    }

class FormsDataset(Dataset):
    def __init__(self, directory, char_to_idx, augmentation=False, img_height=32, only=None):

        self.img_height = img_height
        with open(os.path.join(directory,'transcription.json')) as f:
            types = json.load(f)
        self.lines = []
        if only is None:
            only = types.keys()
        elif type(only) is str:
            only = [only]
        for t in only:
            imgs = types[t]
            for img,trans in imgs.items():
                self.lines.append( (os.path.join(directory,t,img),trans) )


        self.char_to_idx = char_to_idx
        self.augmentation = augmentation
        self.warning=False



    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        img_path, gt = self.lines[idx]

        img = cv2.imread(img_path,0)

        if img is None:
            return None

        if img.shape[0] != self.img_height:
            if img.shape[0] < self.img_height and not self.warning:
                self.warning = True
                print("WARNING: upsampling image to fit size")
            percent = float(self.img_height) / img.shape[0]
            img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)

        if img is None:
            return None

        if self.augmentation:
            img = augmentation.apply_random_color_rotation(img)
            img = augmentation.apply_tensmeyer_brightness(img)
            img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = img / 128.0 - 1.0
        img = img[...,None]

        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
        }
