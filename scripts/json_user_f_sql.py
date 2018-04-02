#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:44 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_user_f_sql.py

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

        # Parse the JSON object in your database
        i = 0
        while data['friends'] is not None and i < len(data['friends']):
            cur = db_conn.cursor()
            cur.execute("insert into user_friends (user_id, friend_user_id) values(%s, %s)",
                        (data['user_id'], data['friends'][i]))
            db_conn.commit()
            # print(i)
            i += 1
        print data['friends']
# Close database connection
db_conn.close()