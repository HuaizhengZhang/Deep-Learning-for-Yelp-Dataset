#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:19 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_business_c_sql.py

import psycopg2

# Import Python's default JSON encoder and decoder
import json

# Open a connection to the database
db_conn = psycopg2.connect("dbname='yelp_db' host='localhost' user='yelp13' password='123456'")

# Initialize an empty list for temporarily holding a JSON object
data = []

# Open a JSON file from the dataset and read data line-by-line iteratively
with open('../dataset/dataset/business.json') as fileobject:
    for line in fileobject:
        data = json.loads(line)

        # Parse the JSON object in your database
        i = 0
        while data['categories'] is not None and i < len(data['categories']):
            cur = db_conn.cursor()
            cur.execute("insert into business_category (business_id, category) values(%s, %s)",
                        (data['business_id'], data['categories'][i]))
            db_conn.commit()
            # print(data['business_id'])
            i += 1
        print data['business_id']

# Close database connection
db_conn.close()