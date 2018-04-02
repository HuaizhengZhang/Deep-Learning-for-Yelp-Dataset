#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 1/4/18 1:41 AM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_user_sql.py

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

        # Commit the data in database
        cur = db_conn.cursor()
        cur.execute(
            "insert into _User (user_id, name, review_count, yelping_since, friends, useful, funny, cool, fans, elite, average_stars, compliment_hot, compliment_more, compliment_profile, compliment_cute, compliment_list, compliment_note, compliment_plain, compliment_cool, compliment_funny, compliment_writer, compliment_photos) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
            data['user_id'], data['name'], data['review_count'], data['yelping_since'], data['friends'], data['useful'],
            data['funny'], data['cool'], data['fans'], data['elite'], data['average_stars'], data['compliment_hot'],
            data['compliment_more'], data['compliment_profile'], data['compliment_cute'], data['compliment_list'],
            data['compliment_note'], data['compliment_plain'], data['compliment_cool'], data['compliment_funny'],
            data['compliment_writer'], data['compliment_photos']))
        db_conn.commit()
        print(data['user_id'])

# Close database connection
db_conn.close()