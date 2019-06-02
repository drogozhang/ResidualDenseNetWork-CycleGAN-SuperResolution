# -*- coding:utf-8 -*-
#
# Created by Drogo Zhang
#
# On 2019-06-01
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import config as cfg


class make_dense(nn.Module):
    def __init__(self, nChannels, growthRate, kernel_size=3):
        super(make_dense, self).__init__()
        self.conv = nn.Conv2d(nChannels, growthRate, kernel_size=kernel_size, padding=(kernel_size - 1) // 2,
                              bias=False)

    def forward(self, x):
        out = F.relu(self.conv(x))
        out = torch.cat((x, out), 1)
        return out


# Residual dense block (RDB) architecture
class RDB(nn.Module):
    def __init__(self, nChannels, nDenselayer, growthRate):
        super(RDB, self).__init__()
        nChannels_ = nChannels
        modules = []
        for i in range(nDenselayer):
            modules.append(make_dense(nChannels_, growthRate))
            nChannels_ += growthRate
        self.dense_layers = nn.Sequential(*modules)
        self.conv_1x1 = nn.Conv2d(nChannels_, nChannels, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        out = self.dense_layers(x)
        out = self.conv_1x1(out)
        out = out + x
        return out


# Residual Dense Network
class H2L_RDN(nn.Module):
    def __init__(self):
        super(H2L_RDN, self).__init__()
        input_channel = cfg.input_channel
        nDenselayer = cfg.nDenselayer
        nFeat = cfg.nFeat
        scale = cfg.scale
        growthRate = cfg.growthRate

        # F-1
        self.conv1 = nn.Conv2d(input_channel, nFeat, kernel_size=3, padding=1, bias=True)
        # F0
        self.conv2 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)
        # RDBs 3
        self.RDB1 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB2 = RDB(nFeat, nDenselayer, growthRate)
        self.RDB3 = RDB(nFeat, nDenselayer, growthRate)
        # global feature fusion (GFF)
        self.GFF_1x1 = nn.Conv2d(nFeat * 3, nFeat, kernel_size=1, padding=0, bias=True)
        self.GFF_3x3 = nn.Conv2d(nFeat, nFeat, kernel_size=3, padding=1, bias=True)

        # Downsampler
        if cfg.downsample_method == 'conv':
            self.conv_down = nn.Conv2d(nFeat, nFeat, kernel_size=cfg.scale, stride=cfg.scale, bias=True)
        elif cfg.downsample_method == 'pool':
            self.pool_down = nn.MaxPool2d(kernel_size=scale, stride=scale)
        # conv
        self.conv3 = nn.Conv2d(nFeat, input_channel, kernel_size=3, padding=1, bias=True)

    def forward(self, x):
        # print("input.shape", x.shape)
        F_ = self.conv1(x)
        F_0 = self.conv2(F_)
        F_1 = self.RDB1(F_0)
        F_2 = self.RDB2(F_1)
        F_3 = self.RDB3(F_2)
        FF = torch.cat((F_1, F_2, F_3), 1)
        FdLF = self.GFF_1x1(FF)
        FGF = self.GFF_3x3(FdLF)
        FDF = FGF + F_
        # Downsampler
        if cfg.downsample_method == 'conv':
            us = self.conv_down(FDF)
        elif cfg.downsample_method == 'pool':
            us = self.pool_down(FDF)
        else:
            raise KeyError("No such down sample method!")

        # print("downsample.shape", us.shape)
        output = self.conv3(us)
        # print("output.shape", output.shape)
        return output
