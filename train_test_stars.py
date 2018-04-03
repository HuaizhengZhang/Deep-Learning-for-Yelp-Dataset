#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2/4/18 4:14 PM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : train_test_stars.py

import os
import psycopg2
import pandas as pd
import langdetect
import matplotlib.pyplot as plt
import pickle


count = 0

def safe_detect(text):
    try:
        global count
        count += 1
        if count % 10000 == 0:
            print(count)
        return langdetect.detect(text)
    except:
        return 'unknown'


def data_prepare():
    db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")
    cur = db_conn.cursor()

    cur.execute(
        "select r1.review_text, r1.stars from review as r1;")

    temp_data = cur.fetchall()
    temp_data = pd.DataFrame(temp_data)

    temp_data.rename(
        columns={temp_data.columns[0]: 'content', temp_data.columns[1]: 'stars'}, inplace=True)

    data = temp_data[:1000000]

    data['language'] = data.content.apply(safe_detect)
    all_data = data[data.language == 'en']
    with open('dataset/reviews.pkl', 'wb') as f:
        pickle.dump(all_data, f)

    print all_data


if __name__ == '__main__':
    data_prepare()
