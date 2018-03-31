#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/3/18 9:39 PM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : functions.py


import os
from torchtext import vocab
from collections import Counter


def get_dicts(words=['hello', 'world'], glove='glove.6B.50d'):
    c = Counter(words)
    v = vocab.Vocab(c, vectors=glove)
    dicts = {}
    for i in words:
        dicts[i] = v.vectors.numpy()[v.stoi[i]]
    return dicts


if __name__ == '__main__':
    get_dicts()