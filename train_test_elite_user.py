#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 4:39 PM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test_elite_user.py

import os
import pandas as pd
import numpy as np
import psycopg2
import json
import torch
import torch.optim as optim
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from DeepCI.nets import *
from DeepCI.data_loader import *


def data_prepare():
    db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")
    cur = db_conn.cursor()

    cur.execute(
        "select u1.user_id, u1.review_count, u1.funny, u1.useful, u1.cool, u1.fans, t1.friend_count, \
        t1.tips_count, u1.elite from _user as u1 join secondary_stats_1 as t1 on u1.user_id= t1.user_id;")
    feature_group1 = cur.fetchall()
    df_feature1 = pd.DataFrame(feature_group1)
    df_feature1.rename(columns={df_feature1.columns[0]: 'user_id'}, inplace=True)

    cur.execute(
        "select t.user_id, count(t.review_id), sum(t.stars), sum(t.review_len) from (select user_id, \
        review_id, stars, length(review_text) as review_len from review) as t group by t.user_id;")
    feature_group2 = cur.fetchall()
    df_feature2 = pd.DataFrame(feature_group2)

    df_feature2[4] = df_feature2[3] / df_feature2[1]
    df_feature2[5] = df_feature2[2] / df_feature2[1]

    df_feature2.rename(columns={df_feature2.columns[0]: 'user_id'}, inplace=True)

    df_all = pd.merge(df_feature1, df_feature2, on='user_id')



    df_all.rename(
        columns={df_all.columns[0]: 'user_id', df_all.columns[1]: 'review_count', df_all.columns[2]: 'funny',
                 df_all.columns[3]: 'useful', df_all.columns[4]: 'cool', df_all.columns[5]: 'fans',
                 df_all.columns[6]: 'friend_count', df_all.columns[7]: 'tips_count', df_all.columns[8]: 'elite',
                 df_all.columns[9]: 'review_count', df_all.columns[12]: 'avg_review_len',
                 df_all.columns[13]: 'avg_rating'}, inplace=True)

    # df_all.elite[df_all.elite != '{}'] = 'Elite'
    # df_all.elite[df_all.elite == '{}'] = 'Not elite'
    # df_all['elite'].replace(df_all['elite'] != '{}', 'Elite', inplace=True)
    # df_all['elite'].replace(df_all['elite'] == '{}', 'Not elite', inplace=True)
    df_all.loc[df_all['elite'] != '{}', 'elite'] = 'Elite'
    df_all.loc[df_all['elite'] == '{}', 'elite'] = 'Not Elite'
    print df_all['elite']

    features = ['review_count', 'avg_review_len', 'avg_rating', 'funny', 'useful', 'cool', 'friend_count',
                'tips_count', 'fans']

    # df_all.elite.value_counts().plot(kind='bar')
    # plt.xlabel('Elite or not')
    # plt.show()

    train_data, test_data = train_test_split(df_all, test_size=0.2)

    train_y = pd.factorize(train_data['elite'])[0]

    train_x = train_data[features]

    test_y = pd.factorize(test_data['elite'])[0]

    test_x = test_data[features]

    min_max_scaler = preprocessing.MinMaxScaler()

    X_train = min_max_scaler.fit_transform(train_x.values)
    X_test = min_max_scaler.transform(test_x.values)

    le = preprocessing.LabelEncoder()
    le.fit(train_y)

    Y_train = le.transform(train_y)
    Y_test = le.transform(test_y)

    print X_train
    print Y_train

    return X_train, Y_train, X_test, Y_test



def train(train_loader, model, optimizer, epoch):
    model.train()
    for batch_idx, sample_batched in enumerate(train_loader):
        pid = os.getpid()

        x = torch.autograd.Variable(sample_batched['features'].cuda())
        target = torch.autograd.Variable(sample_batched['label'].cuda())

        optimizer.zero_grad()

        prediction = model(x)
        # print prediction
        # print target
        loss = F.nll_loss(prediction, target)
        loss.backward()

        optimizer.step()

        if batch_idx % 3 == 0:
            print('{}\tTrain Epoch: {} \tLoss: {:.6f}'.format(
                pid, epoch, loss.data[0]))


def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    for sample_batched in test_loader:
        x = torch.autograd.Variable(sample_batched['features'].cuda())
        target = torch.autograd.Variable(sample_batched['label'].cuda())

        output = model(x)
        test_loss += F.nll_loss(output, target, size_average=False).data[0]
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

def main():
    model = ElitesNN()
    model.cuda()
    print model
    optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.95, eps=1e-08, weight_decay=1e-6)

    x_train, y_train, x_test, y_test = data_prepare()

    train_data = EliteDataset(x_train, y_train)
    train_loader = data_utils.DataLoader(train_data, batch_size=1024, shuffle=False)

    test_data = EliteDataset(x_test, y_test)
    test_loader = data_utils.DataLoader(test_data, batch_size=1024, shuffle=False)

    for epoch in range(30):
        train(train_loader, model, optimizer, epoch)

    test(test_loader, model)


if __name__ =='__main__':
    main()
