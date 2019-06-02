# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-01

import math
import time
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
import argparse
import numpy as np
from torch.nn import init
import torch.optim as optim

from model.CycleGAN import CycleGANModel
from data import DIV2K
from utils import *
import config as cfg


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.kaiming_normal(m.weight.data)


def get_dataset():
    data_train = DIV2K()
    dataloader = torch.utils.data.DataLoader(data_train, batch_size=cfg.batchSize,
                                             drop_last=True, shuffle=True, num_workers=int(cfg.nThreads),
                                             pin_memory=False)
    return dataloader


def set_loss():
    lossType = cfg.lossType
    if lossType == 'MSE':
        lossfunction = nn.MSELoss()
    elif lossType == 'L1':
        lossfunction = nn.L1Loss()
    else:
        raise KeyError('No such Loss Function Defined')
    return lossfunction


def set_lr(epoch, optimizer):
    lrDecay = cfg.lrDecay
    decayType = cfg.decayType
    if decayType == 'step':
        epoch_iter = (epoch + 1) // lrDecay
        lr = cfg.lr / 2 ** epoch_iter
    elif decayType == 'exp':
        k = math.log(2) / lrDecay
        lr = cfg.lr * math.exp(-k * epoch)
    elif decayType == 'inv':
        k = 1 / lrDecay
        lr = cfg.lr / (1 + k * epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train():
    #  select network
    cycle_gan = CycleGANModel()
    print(cycle_gan)
    cycle_gan.initialize()
    cycle_gan.cuda()

    cycle_gan.setup()
    log = LossLog()
    # fine-tuning or retrain
    if cfg.continue_train:
        cycle_gan = cycle_gan.load_networks(cfg.which_epoch)
    # load data
    dataloader = get_dataset()

    total_steps = 0
    for epoch in range(cfg.epochs):
        iter_data_time = time.time()
        epoch_iter = 0
        for batch, (im_lr, im_hr) in enumerate(dataloader):
            iter_start_time = time.time()
            if total_steps % cfg.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            img_low_resolution = Variable(im_lr.cuda(), volatile=False)
            img_high_resolution = Variable(im_hr.cuda())
            input_dict = {'A': img_low_resolution,
                          'B': img_high_resolution}

            total_steps += cfg.batchSize
            epoch_iter += cfg.batchSize
            cycle_gan.set_input(input_dict)
            cycle_gan.optimize_parameters()

            if total_steps % cfg.print_freq == 0:
                losses = cycle_gan.get_current_losses()
                t = (time.time() - iter_start_time) / cfg.batchSize
                log.print_current_losses(epoch, epoch_iter, losses, t, t_data)
        if epoch % cfg.save_epoch_freq == 0:
            cycle_gan.save_networks(epoch)


if __name__ == '__main__':
    train()
