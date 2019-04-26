import json

import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

from collections import defaultdict
import os
import cv2
import numpy as np
import math, random

from . import grid_distortion

from utils import string_utils, safe_load, augmentation
#from utils import SyntheticText
from synthetic_text_gen import SyntheticText
#from utils.generator_net import define_G

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

class SyntheticDataset(Dataset):
    def __init__(self, char_to_idx, augmentation=False, img_height=32, param_file=None, generator_aux_path=None, dataset_size=1000,cuda=True):
        self.dataset_size=dataset_size #fake, but how many to run during validation
        self.img_height = img_height
        
        #self.synthetic = SyntheticText('../data/text_fonts','../data/OANC_text',line_prob=0.8,line_thickness=70,line_var=30,pad=20,gaus_noise=0.15,hole_prob=0.6, hole_size=400,neighbor_gap_var=30,rot=2.5,text_len=40)
        self.synthetic = []
        with open(param_file) as f:
            syn_params = json.load(f)
        for gen in syn_params:
            self.synthetic.append(SyntheticText(
                gen['font_dir'],
                gen['text_dir'],
                line_prob=gen['line_prob'],
                line_thickness=gen['line_thickness'],
                line_var=gen['line_var'],
                pad=gen['pad'],
                gaus_noise=gen['gaus_noise'],
                hole_prob=gen['hole_prob'],
                hole_size=gen['hole_size'],
                neighbor_gap_var=gen['neighbor_gap_var'],
                rot=gen['rot'],
                text_len=gen['text_len'],
                use_warp=gen['use_warp'],
                warp_std=gen['warp_std'],
                warp_intr=gen['warp_intr']
                ))
        if generator_aux_path is not None:
            aux = torch.load(generator_aux_path)
            for i,prob in enumerate(aux['font_prob']):
                self.synthetic[i].fontProbs=prob
        #opt={   'output_nc':1,
        #        'input_nc':1,
        #        'ngf':64,
        #        'netG':'resnet_6blocks',
        #        'norm': 'instance',
        #        'no_dropout': True,
        #        'init_type': 'normal',
        #        'init_gain': 0.02,
        #        'gpu_ids': []
        #        }
        #self.generator = define_G(opt['output_nc'], opt['input_nc'], opt['ngf'], opt['netG'], opt['norm'],
        #                                not opt['no_dropout'], opt['init_type'], opt['init_gain'], opt['gpu_ids'])
        #state_dict = torch.load(generator_load_path)#, map_location=str(self.generator.module.model[1].weight.device))
        #self.generator.load_state_dict(state_dict)
        #self.generator.eval()
        #for param in self.generator.parameters():
        #    param.requires_grad = False

        #self.cuda=cuda
        #if cuda:
        #    self.generator=self.generator.cuda()

        self.char_to_idx = char_to_idx
        self.augmentation = augmentation and not gen['use_warp']
        self.warning=False



    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        syn_gen = random.choice(self.synthetic)
        syn_img, gt, f_index = syn_gen.getSample()

        w=round(syn_img.shape[1] * float(self.img_height)/syn_img.shape[0])
        img = cv2.resize(syn_img,(w,self.img_height),interpolation = cv2.INTER_CUBIC)

        #syn_img=torch.from_numpy(syn_img).float()[None,None,...]
        #if self.cuda:
        #    syn_img = syn_img.cuda()
        #img = self.generator(syn_img)

        #if img.shape[0] != self.img_height:
        #    if img.shape[0] < self.img_height and not self.warning:
        #        self.warning = True
        #        print "WARNING: upsampling image to fit size"
        #    percent = float(self.img_height) / img.shape[0]
        #    img = cv2.resize(img, (0,0), fx=percent, fy=percent, interpolation = cv2.INTER_CUBIC)
        #img = (img.cpu().numpy()[0,0,...]+1)*128
        img*=255
        img = img.astype(np.uint8)

        if self.augmentation:
            #img = augmentation.apply_random_color_rotation(img)
            #img = augmentation.apply_tensmeyer_brightness(img) done in sythetic text generator
            if random.random()<0.5:
                img = grid_distortion.warp_image(img)

        img = img.astype(np.float32)
        img = (img / 128.0 - 1.0)[...,None]
        if len(gt) == 0:
            return None
        gt_label = string_utils.str2label_single(gt, self.char_to_idx)


        return {
            "line_img": img,
            "gt": gt,
            "gt_label": gt_label
            }
