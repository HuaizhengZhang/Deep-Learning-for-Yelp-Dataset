#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 31/3/18 11:36 PM
# @Author  : Huaizheng ZHANG
# @Site    : zhanghuaizheng.info
# @File    : json_business_sql.py

# Parse listed businesses in a Postgres database with all fields, except categories
# Version 1.0
# Author: Manoj Pravakar Saha
# This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.

# Import Psycopg Postgres adapter for Python
import psycopg2

# Import Python's default JSON encoder and decoder
import json

# Open a connection to the database
db_conn = psycopg2.connect("dbname='yelp_13' host='localhost' user='yelp13' password='123456'")

# Initialize an empty list for temporarily holding a JSON object
data = []

# Open a JSON file from the dataset and read data line-by-line iteratively
with open('../dataset/dataset/business.json') as fileobject:
    for line in fileobject:
        data = json.loads(line)

        # Parse the JSON object in your database
        cur = db_conn.cursor()
        cur.execute(
            "insert into Business (business_id, name, neighborhood, address, city, state, postal_code, latitude, longitude, stars, review_count, is_open, attributes, hours, type) values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (data['business_id'], data['name'], data['neighborhood'], data['address'], data['city'], data['state'],
             data['postal_code'], data['latitude'], data['longitude'], data['stars'], data['review_count'],
             data['is_open'], data['attributes'], data['hours'], data['type']))
        db_conn.commit()
        # print(data['user_id'])

# Close database connection
db_conn.close()