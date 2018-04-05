# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:34:45 2018

@author: jaide
"""

import sqlite3 
import pandas as pd

conn = sqlite3.connect("C:\\Users\\jaide\\Downloads\\soccer\\database.sqlite")

cur=conn.cursor()

cur.execute("select * from Player_Attributes limit 5").fetchall()

results=cur.fetchall()

print(results)


#------------------------
table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
print(table['name'])
df={}
for n in table['name']:
    print(n)
    #df[n]=pd.DataFrame()
    df[n]=pd.read_sql_query("SELECT * from %s" % n, conn)
    df[n].to_csv(n + '.csv', index_label='index')
    
print(df['Team'].head(2))    
#locals().update(df) ------->>> is messing with local namespace a bad idea?? research needed
#Team.head(5)
type(df)
