#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/4/18 12:57 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : nets.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class ElitesNN(nn.Module):
    def __init__(self):
        super(ElitesNN, self).__init__()
        self.input = nn.Linear(10, 512)

        self.fc1 = nn.Linear(512, 1024)
        self.bn_1 = nn.BatchNorm2d(1024)

        self.fc2 = nn.Linear(1024, 256)
        self.bn_2 = nn.BatchNorm2d(256)

        self.fc3 = nn.Linear(256, 2)

    def forward(self, x1):
        h = self.input(x1)
        h = F.tanh(h)

        h = F.tanh(self.bn_1(self.fc1(h)))
        h = F.dropout(h, p=0.5, training=self.training)

        fc2 = F.tanh(self.bn_2(self.fc2(h)))
        h = F.dropout(fc2, p=0.5, training=self.training)

        h = F.log_softmax(self.fc3(h))
        return h