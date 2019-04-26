import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from warpctc_pytorch import CTCLoss

from hw import hw_dataset
from hw import synthetic_dataset
from hw import cnn_lstm
from hw.forms_dataset import FormsDataset
from hw.synthetic_dataset import SyntheticDataset

from utils.dataset_wrapper import DatasetWrapper
from utils import safe_load

import numpy as np
import cv2
import sys
import json
import os
from utils import string_utils, error_rates
import time
import random
import yaml
from utils.generator_net import define_G

from utils.dataset_parse import load_file_list

cuda=True

with open(sys.argv[1]) as f:
    config = yaml.load(f)

hw_network_config = config['network']['hw']
pretrain_config = config['pretraining']

char_set_path = hw_network_config['char_set_path']

threads = hw_network_config['num_threads']

with open(char_set_path) as f:
    char_set = json.load(f)

idx_to_char = {}
for k,v in char_set['idx_to_char'].items():
    idx_to_char[int(k)] = v

generator_prefix = pretrain_config['generator_path']
generator_load_path = generator_prefix+'_net_G_B.pth'
generator_aux_path = generator_prefix+'_aux.pth'
train_dataset = SyntheticDataset(
                          char_set['char_to_idx'], augmentation=True,
                          img_height=hw_network_config['input_height'],
                          param_file = pretrain_config['synth_params'],
                          generator_aux_path = generator_aux_path,
                          dataset_size=12000)

train_dataloader = DataLoader(train_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=True, num_workers=threads, drop_last=True,
                             collate_fn=hw_dataset.collate)

#batches_per_epoch = int(len(train_dataloader)/pretrain_config['hw']['batch_size'])
#train_dataloader = DatasetWrapper(train_dataloader, batches_per_epoch)

test_set_list = load_file_list(pretrain_config['validation_set']) if 'validation_set' in pretrain_config else None
if test_set_list is not None:
    test_dataset = FormsDataset(test_set_list,
                             char_set['char_to_idx'],
                             img_height=hw_network_config['input_height'])
else:
    test_dataset = SyntheticDataset(
                             char_set['char_to_idx'],
                             img_height=hw_network_config['input_height'],
                             param_file = pretrain_config['synth_params'],
                             generator_aux_path = generator_aux_path,
                             dataset_size=2400)

test_dataloader = DataLoader(test_dataset,
                             batch_size=pretrain_config['hw']['batch_size'],
                             shuffle=False, num_workers=threads,
                             collate_fn=hw_dataset.collate)



criterion = CTCLoss()

hw = cnn_lstm.create_model(hw_network_config)
load_path = os.path.join(pretrain_config['snapshot_path'], '{}_latest.pt'.format(pretrain_config['snapshot_prefix']))
if os.path.isfile(load_path):
    loaded = torch.load(load_path)
    
    hw.load_state_dict(loaded['state_dict'])
    lowest_cer = loaded['lowest_cer']
    log = loaded['log']
    start_epoch = loaded['epoch']+1
    print('Loaded at epoch {}'.format(start_epoch))
else:
    lowest_cer = np.inf
    log=[]
    start_epoch=0


opt={   'output_nc':1,
        'input_nc':1,
        'ngf':64,
        'netG':'resnet_6blocks',
        'norm': 'instance',
        'no_dropout': True,
        'init_type': 'normal',
        'init_gain': 0.02,
        'gpu_ids': []
        }
generator = define_G(opt['output_nc'], opt['input_nc'], opt['ngf'], opt['netG'], opt['norm'],
                                not opt['no_dropout'], opt['init_type'], opt['init_gain'], opt['gpu_ids'])
state_dict = torch.load(generator_load_path)#, map_location=str(self.generator.module.model[1].weight.device))
generator.load_state_dict(state_dict)
generator.eval()
for param in generator.parameters():
    param.requires_grad = False

if cuda:
    generator=generator.cuda()
    hw=hw.cuda()

optimizer = torch.optim.Adam(hw.parameters(), lr=pretrain_config['hw']['learning_rate'])
dtype = torch.cuda.FloatTensor

cnt_since_last_improvement = 0 
print_freq=50
for epoch in range(start_epoch,1000):
    print("Epoch {}".format(epoch))
    steps = 0.0
    hw.train()
    sum_loss=0
    sum_cer = 0.0
    for i, x in enumerate(train_dataloader):
        print('iteration: {} / {}'.format(i,len(train_dataloader)), end='\r')

        line_imgs = x['line_imgs'].type(dtype)
        labels =  x['labels']
        label_lengths = x['label_lengths']

        line_imgs = generator(line_imgs)
        #for b in range(line_imgs.size(0)):
        #    draw = ((line_imgs[b,0]+1)*128).cpu().numpy().astype(np.uint8)
        #    cv2.imwrite('test/line{}.png'.format(b),draw)
        #    print('gt[{}]: {}'.format(b,x['gt'][b]))
        #cv2.waitKey()
        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()
        toprint=[]
        for b, gt_line in enumerate(x['gt']):
            logits = out[b,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_cer += cer
            steps += 1
            
            if i%print_freq==0:
                toprint.append('[cer]:{:.2f} [gt]: {} [pred]: {}'.format(cer, gt_line,pred_str))



        batch_size = preds.size(1)
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)

        # print "before"
        loss = criterion(preds, labels, preds_size, label_lengths)
        # print "after"

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss = loss.item()/line_imgs.size(3)
        sum_loss += loss
        if i%print_freq==0:
            print('iteration: {}, loss: {}'.format(i,loss))
            for line in toprint:
                print(line)

    tolog={'train_loss':loss,'train_cer':sum_cer/steps}
    print("Train Loss: {}, CER: {}".format( sum_loss/len(train_dataloader),sum_cer/steps))
    #print("Real Epoch {}".format(train_dataloader.epoch))

    sum_cer = 0.0
    steps = 0.0
    hw.eval()

    for i,x in enumerate(test_dataloader):
        line_imgs = x['line_imgs'].type(dtype)
        labels =  x['labels']
        label_lengths = x['label_lengths']

        if test_set_list is None:
            line_imgs=generator(line_imgs)

        preds = hw(line_imgs).cpu()

        output_batch = preds.permute(1,0,2)
        out = output_batch.data.cpu().numpy()

        if i==0:
            print('Test examples:')
        for b, gt_line in enumerate(x['gt']):
            logits = out[b,...]
            pred, raw_pred = string_utils.naive_decode(logits)
            pred_str = string_utils.label2str_single(pred, idx_to_char, False)
            cer = error_rates.cer(gt_line, pred_str)
            sum_cer += cer
            steps += 1
            #draw = ((line_imgs[i,0]+1)*128).cpu().numpy().astype(np.uint8)
            #cv2.imwrite('test/line{}.png'.format(i),draw)
            if i==0:
                print('[cer]:{:.2f} [gt]: {} [pred]: {}'.format(cer, gt_line,pred_str))

    cnt_since_last_improvement += 1
    tolog['test_cer'] = sum_cer/steps
    log.append(tolog)
    tosave={
            'log':log,
            'epoch':epoch,
            'state_dict':hw.state_dict(),
            'lowest_cer':lowest_cer
            }
    if lowest_cer >= sum_cer/steps:
        cnt_since_last_improvement = 0
        lowest_cer = sum_cer/steps
        print("Saving Best")

        if not os.path.exists(pretrain_config['snapshot_path']):
            os.makedirs(pretrain_config['snapshot_path'])

        torch.save(tosave, os.path.join(pretrain_config['snapshot_path'], '{}_best.pt'.format(pretrain_config['snapshot_prefix'])))
    torch.save(tosave, os.path.join(pretrain_config['snapshot_path'], '{}_latest.pt'.format(pretrain_config['snapshot_prefix'])))

    print("Test CER: {}, best: {}".format(sum_cer/steps, lowest_cer))
    print()

    if cnt_since_last_improvement >= pretrain_config['hw']['stop_after_no_improvement'] and lowest_cer<0.9:
        break
