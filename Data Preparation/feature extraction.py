# -*- coding: utf-8 -*-
"""
Created on Thu Apr 2 18:22:57 2018

@author: pragy
"""

import sqlite3 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

conn = sqlite3.connect("F:\\DPA\\Project\\database.sqlite")
cur=conn.cursor()

#------------------------------------------------------------------------------------------------------



table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
df={}
for n in table['name']:
    print(n)
    df[n]=pd.read_sql_query("SELECT * from %s" % n, conn)
    df[n].to_csv(n + '.csv', index_label='index')
    
#locals().update(df) ------->>> is messing with local namespace a bad idea?? research needed
print(df.keys())

#retrieves year portion of a date
def get_year(x):
    m= x.split(" ")[0]
    m = datetime.datetime.strptime(m, "%Y-%m-%d").date()
    return int(m.year)

#Extracts year from "date" column of "Player_Attributes" and adds it as an additional column "year"
year =[]
for i in range(0, len(df['Player_Attributes'])):
    try:
        year.append(get_year(df['Player_Attributes'].date[i]))
        
    except:
        year.append(0)
        pass

len(year)
df['Player_Attributes']['year']=year
df['Player_Attributes'].year = year
df['Player_Attributes'].year[20]
len(df['Player_Attributes'].shape)


#Extracts year from "date" column of "Match" and adds it as an additional column "year"
year_match =[]
for i in range(0, len(df['Match'])):
    try:
        year_match.append(get_year(df['Match'].date[i]))
        
    except:
        year_match.append(0)
        pass
len(year_match)
df['Match']['year'] = year_match
df['Match'].year = year_match
df['Match'].head(1)
df['Match'].year[20]

#Evaluate the mean y-axis for each player based on match lineups
def get_position(x):
   
    data=df['Match'][(df['Match'].year >= 2008) & (df['Match'].year <= 2010)]
    print(len(data))
    data1=df['Match'][(df['Match'].year > 2010) & (df['Match'].year <= 2013)]
    print(len(data1))
    data2=df['Match'][(df['Match'].year > 2013) & (df['Match'].year <= 2016)]
    print(len(data2))
         
    position=[]
    print("calculating for 2008-2010 bucket")
    
    d=data.home_player_Y1[data.home_player_1==x]
    d=d.append(data.home_player_Y2[data.home_player_2==x])
    d=d.append(data.home_player_Y3[data.home_player_3==x])
    d=d.append(data.home_player_Y4[data.home_player_4==x])
    d=d.append(data.home_player_Y5[data.home_player_5==x])
    d=d.append(data.home_player_Y6[data.home_player_6==x])
    d=d.append(data.home_player_Y7[data.home_player_7==x])
    d=d.append(data.home_player_Y8[data.home_player_8==x])
    d=d.append(data.home_player_Y9[data.home_player_9==x])
    d=d.append(data.home_player_Y10[data.home_player_10==x])
    d=d.append(data.home_player_Y11[data.home_player_11==x])
    
    d=d.append(data.away_player_Y1[data.away_player_1==x])
    d=d.append(data.away_player_Y2[data.away_player_2==x])
    d=d.append(data.away_player_Y3[data.away_player_3==x])
    d=d.append(data.away_player_Y4[data.away_player_4==x])
    d=d.append(data.away_player_Y5[data.away_player_5==x])
    d=d.append(data.away_player_Y6[data.away_player_6==x])
    d=d.append(data.away_player_Y7[data.away_player_7==x])
    d=d.append(data.away_player_Y8[data.away_player_8==x])
    d=d.append(data.away_player_Y9[data.away_player_9==x])
    d=d.append(data.away_player_Y10[data.away_player_10==x])
    d=d.append(data.away_player_Y11[data.away_player_11==x])
    
    
    if len(d) > 0:
                Y = np.array(d,dtype=np.float)
                mean_y = np.nanmean(Y)
                position.append(mean_y)
    else:
        position.append(0)
             
    
    print(position)
    print(" ")
    print("calculating for 2010-2013 bucket")
    
    d1=data1.home_player_Y1[data1.home_player_1==x]
    d1=d1.append(data1.home_player_Y2[data1.home_player_2==x])
    d1=d1.append(data1.home_player_Y3[data1.home_player_3==x])
    d1=d1.append(data1.home_player_Y4[data1.home_player_4==x])
    d1=d1.append(data1.home_player_Y5[data1.home_player_5==x])
    d1=d1.append(data1.home_player_Y6[data1.home_player_6==x])
    d1=d1.append(data1.home_player_Y7[data1.home_player_7==x])
    d1=d1.append(data1.home_player_Y8[data1.home_player_8==x])
    d1=d1.append(data1.home_player_Y9[data1.home_player_9==x])
    d1=d1.append(data1.home_player_Y10[data1.home_player_10==x])
    d1=d1.append(data1.home_player_Y11[data1.home_player_11==x])
    
    d1=d1.append(data1.away_player_Y1[data1.away_player_1==x])
    d1=d1.append(data1.away_player_Y2[data1.away_player_2==x])
    d1=d1.append(data1.away_player_Y3[data1.away_player_3==x])
    d1=d1.append(data1.away_player_Y4[data1.away_player_4==x])
    d1=d1.append(data1.away_player_Y5[data1.away_player_5==x])
    d1=d1.append(data1.away_player_Y6[data1.away_player_6==x])
    d1=d1.append(data1.away_player_Y7[data1.away_player_7==x])
    d1=d1.append(data1.away_player_Y8[data1.away_player_8==x])
    d1=d1.append(data1.away_player_Y9[data1.away_player_9==x])
    d1=d1.append(data1.away_player_Y10[data1.away_player_10==x])
    d1=d1.append(data1.away_player_Y11[data1.away_player_11==x])
    
    #print(d1)
    if len(d1) > 0:
                Y = np.array(d1,dtype=np.float)
                mean_y1 = np.nanmean(Y)
                position.append(mean_y1)
    else:
        position.append(0)
         
    
    print(position)
    print(" ")
    print("calculating for 2013-2016 bucket")
    
    d2=data2.home_player_Y1[data2.home_player_1==x]
    d2=d2.append(data2.home_player_Y2[data2.home_player_2==x])
    d2=d2.append(data2.home_player_Y3[data2.home_player_3==x])
    d2=d2.append(data2.home_player_Y4[data2.home_player_4==x])
    d2=d2.append(data2.home_player_Y5[data2.home_player_5==x])
    d2=d2.append(data2.home_player_Y6[data2.home_player_6==x])
    d2=d2.append(data2.home_player_Y7[data2.home_player_7==x])
    d2=d2.append(data2.home_player_Y8[data2.home_player_8==x])
    d2=d2.append(data2.home_player_Y9[data2.home_player_9==x])
    d2=d2.append(data2.home_player_Y10[data2.home_player_10==x])
    d2=d2.append(data2.home_player_Y11[data2.home_player_11==x])
    
    d2=d2.append(data2.away_player_Y1[data2.away_player_1==x])
    d2=d2.append(data2.away_player_Y2[data2.away_player_2==x])
    d2=d2.append(data2.away_player_Y3[data2.away_player_3==x])
    d2=d2.append(data2.away_player_Y4[data2.away_player_4==x])
    d2=d2.append(data2.away_player_Y5[data2.away_player_5==x])
    d2=d2.append(data2.away_player_Y6[data2.away_player_6==x])
    d2=d2.append(data2.away_player_Y7[data2.away_player_7==x])
    d2=d2.append(data2.away_player_Y8[data2.away_player_8==x])
    d2=d2.append(data2.away_player_Y9[data2.away_player_9==x])
    d2=d2.append(data2.away_player_Y10[data2.away_player_10==x])
    d2=d2.append(data2.away_player_Y11[data2.away_player_11==x])
    
    #print(d2)
    if len(d2) > 0:
                Y = np.array(d2,dtype=np.float)
                mean_y2 = np.nanmean(Y)
                position.append(mean_y2)
    else:
        position.append(0)

                    
    
    print(position)
        
    return position
    
    
        
#Test: (uncomment any below lines to test the function)
#get_position(36835)    
#get_position(38788)
#get_position(94462)
#get_position(37069)
#get_position(50160)
  
#calculating bucket810, bucket1013 and bucket1316 for all player IDs
#bucket810: average Y-axis coordinate for all the matches played by the player between 2008 and 2010
#bucket1013: average Y-axis coordinate for all the matches played by the player between 2010 and 2013
#bucket1316: average Y-axis coordinate for all the matches played by the player between 2013 and 2016
pos=[]
bucket810 =[]
bucket1013=[]
bucket1316=[]
for i in range(0,len(df['Player_Attributes'])):
    
    result= get_position(df['Player_Attributes'].player_api_id[i])
    bucket810.append(result[0])
    bucket1013.append(result[1])
    bucket1316.append(result[2])
    
   
bucket1 = bucket810
bucket2= bucket1013
bucket3 = bucket1316
for i in range(0,len(df['Player_Attributes'])):
    if (bucket1[i]==0 or bucket1[i]==None or str(bucket1[i])=='nan'):
        bucket1[i] = np.nan
    if (bucket2[i]==0 or bucket2[i]==None or str(bucket2[i])=='nan'):
        bucket2[i] = np.nan
    if (bucket3[i]==0 or bucket3[i]==None or str(bucket3[i])=='nan'):
        bucket3[i] = np.nan   
    
bucket1 = np.array(bucket1,dtype=float)  
bucket2 = np.array(bucket2,dtype=float)  
bucket3 = np.array(bucket3,dtype=float)  


bucket_mean1=[]
bucket_std1 =[]

for i in range(0,len(bucket3)):
    arr=[]
    arr_diff=[]
    arr.append(bucket1[i])
    arr.append(bucket2[i])
    arr.append(bucket3[i])
    arr=np.array(arr,dtype=np.float)
    arr_mean = np.nanmean(arr)
    bucket_mean1.append(arr_mean)    
    bucket_std1.append(np.nanstd(arr))
    arr_diff=np.array(arr_diff,dtype=np.float)

    
bucket_mean1[0]
bucket_std1[0]


#Adding bucket1, bucket2, bucket3, bucket_mean1, bucket_std1 to "Player_Attributes" 
df['Player_Attributes']['bucket810'] = bucket1
df['Player_Attributes']['bucket1013'] = bucket2
df['Player_Attributes']['bucket1316'] = bucket3
df['Player_Attributes']['bucket_mean1'] = bucket_mean1
df['Player_Attributes']['bucket_std1'] = bucket_std1
df['Player_Attributes'].to_csv('Player_Attributes_new_v22' + '.csv', index_label='index') 

# Grouping and labelling:
bucket_mean = pd.read_csv('F:/Player-Analytics/Player_Attributes_v22.csv')
list(bucket_mean)
bucket_mean1 = np.array(bucket_mean.bucket_mean1, dtype=np.float64)


fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating histogram')
bp=ax.hist(bucket_mean1, bins =16, color="purple")
print('saving figure')
fig.savefig('bucket_mean1_hist.png')
plt.close(fig)


#Using k-means clustering to partition data into 4 parts for each position
len(bucket_mean1.reshape(-1,1))

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(bucket_mean1.reshape(-1,1))
bucket_mean['kmean_predict'] = kmeans.predict(bucket_mean1.reshape(-1,1))

bucket_mean['kmeans_predict_position'] =np.where((bucket_mean['kmean_predict']==0), 'for',bucket_mean['kmean_predict'])
bucket_mean['kmeans_predict_position'] =np.where((bucket_mean['kmean_predict']==1), 'def',bucket_mean['kmeans_predict_position'])
bucket_mean['kmeans_predict_position'] =np.where((bucket_mean['kmean_predict']==2), 'mid',bucket_mean['kmeans_predict_position']) 
bucket_mean['kmeans_predict_position'] =np.where((bucket_mean['kmean_predict']==3), 'gk',bucket_mean['kmeans_predict_position'])

bucket_mean.kmeans_predict_position.value_counts()
bucket_mean.to_csv('F:/Player-Analytics/kmean_predicted_on_bucket_mean.csv')
