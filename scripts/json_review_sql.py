#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:32 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_review_sql.py

import psycopg2

# Import Python's default JSON encoder and decoder
import json

# Open a connection to the database
db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")

# Initialize an empty list for temporarily holding a JSON object
data = []

# Open a JSON file from the dataset and read data line-by-line iteratively
with open('../dataset/dataset/review.json') as fileobject:
    for line in fileobject:
        data = json.loads(line)
        # Parse the JSON object in your database
        cur = db_conn.cursor()
        cur.execute("insert into Review (review_id, user_id, business_id, stars, review_date, review_text, useful, funny, cool) values (%s, %s, %s, %s, %s, %s, %s, %s, %s)", (data['review_id'], data['user_id'], data['business_id'], data['stars'],data['date'],data['text'], data['useful'], data['funny'], data['cool']))
        db_conn.commit()
        #print(data['review_id'])
        print data['user_id']
# Close database connection
db_conn.close()