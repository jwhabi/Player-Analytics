# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:34:45 2018

@author: jaide
"""

import sqlite3 
import pandas as pd
import numpy as np

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
print(df['Match'].keys())

print(df['Player_Attributes'].head(2))
print(df['Player_Attributes']['player_api_id'].head(2))

def get_position(x):
    #data=df['Player_Attributes'][df['Player_Attributes'].player_api_id==x]
    #print(data['player_api_id'])
    d2=df['Match'].home_player_Y1[df['Match'].home_player_1==x]
    d2=d2.append(df['Match'].home_player_Y2[df['Match'].home_player_2==x])
    d2=d2.append(df['Match'].home_player_Y3[df['Match'].home_player_3==x])
    d2=d2.append(df['Match'].home_player_Y4[df['Match'].home_player_4==x])
    d2=d2.append(df['Match'].home_player_Y5[df['Match'].home_player_5==x])
    d2=d2.append(df['Match'].home_player_Y6[df['Match'].home_player_6==x])
    d2=d2.append(df['Match'].home_player_Y7[df['Match'].home_player_7==x])
    d2=d2.append(df['Match'].home_player_Y8[df['Match'].home_player_8==x])
    d2=d2.append(df['Match'].home_player_Y9[df['Match'].home_player_9==x])
    d2=d2.append(df['Match'].home_player_Y10[df['Match'].home_player_10==x])
    d2=d2.append(df['Match'].home_player_Y11[df['Match'].home_player_11==x])
    
    d2=d2.append(df['Match'].away_player_Y1[df['Match'].away_player_1==x])
    d2=d2.append(df['Match'].away_player_Y2[df['Match'].away_player_2==x])
    d2=d2.append(df['Match'].away_player_Y3[df['Match'].away_player_3==x])
    d2=d2.append(df['Match'].away_player_Y4[df['Match'].away_player_4==x])
    d2=d2.append(df['Match'].away_player_Y5[df['Match'].away_player_5==x])
    d2=d2.append(df['Match'].away_player_Y6[df['Match'].away_player_6==x])
    d2=d2.append(df['Match'].away_player_Y7[df['Match'].away_player_7==x])
    d2=d2.append(df['Match'].away_player_Y8[df['Match'].away_player_8==x])
    d2=d2.append(df['Match'].away_player_Y9[df['Match'].away_player_9==x])
    d2=d2.append(df['Match'].away_player_Y10[df['Match'].away_player_10==x])
    d2=d2.append(df['Match'].away_player_Y11[df['Match'].away_player_11==x])
    
    #print(d2)
    if len(d2) > 0:
                Y = np.array(d2,dtype=np.float)
                mean_y = np.nanmean(Y)
                print(mean_y)
                if (mean_y >= 10.0):
                    return "for"
                elif (mean_y > 5):
                    return "mid"
                elif (mean_y > 1):
                    return "def"
                elif (mean_y == 1.0):
                    return "gk"
    return None
        

#get_position(36835)    
#get_position(38788)
#get_position(94462)
#get_position(37069)
#get_position(50160)
pos=[]


for i in range(0,len(df['Player_Attributes'])):
    #df['Player_Attributes']['position'][i]=get_position(df['Player_Attributes'].player_api_id[i])
    pos.append(get_position(df['Player_Attributes'].player_api_id[i]))
    
df['Player_Attributes']['position'] = pos
df['Player_Attributes'].to_csv('Player_Attributes' + '.csv', index_label='index')   
#print(len(pos))
#
#x=[]
#for i in range(0,len(df['Player_Attributes'])):
#    x.append(i)
#print(x)   
#get_position(df['Player_Attributes'].player_api_id[183978]) 
#get_position(df['Player_Attributes'].player_api_id[0])