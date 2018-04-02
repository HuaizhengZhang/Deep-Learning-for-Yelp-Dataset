#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/4/18 12:46 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : data_loader.py

import torch
import numpy as np
from torch.utils.data import Dataset


class EliteDataset(Dataset):
    def __init__(self, x, y):
        self.x, self.y = x, y


    def __len__(self):
        return len(self.x)


    def __getitem__(self, idx):
        features = torch.from_numpy(self.x[idx].astype(np.float)).float()

        label = self.y[idx]

        sample = {"features": features, 'label': label}

        return sample
