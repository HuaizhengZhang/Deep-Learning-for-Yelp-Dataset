#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:47 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_elite_sql.py

import psycopg2

# Import Python's default JSON encoder and decoder
import json

# Open a connection to the database
db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")

# Initialize an empty list for temporarily holding a JSON object
data = []

# Open a JSON file from the dataset and read data line-by-line iteratively
with open('../dataset/dataset/user.json') as fileobject:
    for line in fileobject:
        data = json.loads(line)
        i = 0
        while i < len(data['elite']):
            if data['elite'][i] != "None":
                cur = db_conn.cursor()
                cur.execute("insert into user_elite (user_id, elite_year) values (%s, %s)", (data['user_id'], data['elite'][i]))
                db_conn.commit()
            else:
                break

            i += 1
        print data['elite']


db_conn.close()