#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 3/4/18 2:38 PM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : plot.py

import pickle
import numpy as np
import matplotlib.pyplot as plt

with open('../models/loss.pkl', 'rb') as f:
    y = pickle.load(f)

x = np.arange(len(y))

plt.figure()
plt.plot(x, y)
plt.yscale('linear')
plt.title('Loss')
plt.grid(True)

plt.show()