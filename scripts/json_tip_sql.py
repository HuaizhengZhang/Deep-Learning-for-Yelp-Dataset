#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:37 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_tip_sql.py

import psycopg2
import json

# Load the whole JSON file in memory
#f = open('yelp_academic_dataset_tip.json', 'r')

# Initialize an empty list for temporarily holding JSON objects as list
data = []

# Open a connection to the database
db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")

# Iteratively commit the data from the list in database
with open('../dataset/dataset/tip.json') as fileobject:
    for line in fileobject:
        data = json.loads(line)
        cur = db_conn.cursor()
        cur.execute("insert into Tip (user_id, business_id, tip_text, tip_date, likes) values (%s, %s, %s, %s, %s)", (data['user_id'], data['business_id'], data['text'], data['date'], data['likes']))
        db_conn.commit()
        print data['user_id']



# Close database connection
db_conn.close()