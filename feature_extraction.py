# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 18:22:57 2018

@author: pragy
"""

import sqlite3 
import pandas as pd
import numpy as np
import itertools

import os
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
bucket_total=[]
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



#calculating diff12, diff23 and diff31
diff12= []
diff23=[]
diff31 =[]
for i in range(0,len(bucket1)):

    if ((bucket1[i]==np.nan) and (bucket2[i]== np.nan)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif ((bucket2[i]==np.nan) and (bucket3[i]== np.nan)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif ((bucket1[i]==np.nan) and (bucket3[i]== np.nan)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif (bucket1[i]==np.nan):
        diff12.append(np.nan)
        diff23.append(abs(bucket2[i]-bucket3[i]))
        diff31.append(np.nan)
    elif (bucket2[i]==np.nan):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(abs(bucket1[i]-bucket3[i]))
    elif (bucket3[i]==np.nan):
        diff12.append(abs(bucket1[i]-bucket2[i]))
        diff23.append(np.nan)
        diff31.append(np.nan)
    else:
        diff12.append(abs(bucket1[i]-bucket2[i]))
        diff23.append(abs(bucket2[i]-bucket3[i]))
        diff31.append(abs(bucket3[i]-bucket1[i])) 
        
 
diff1 = np.array(diff12, dtype=float) 
diff2 = np.array(diff23, dtype=float)
diff3 = np.array(diff31, dtype=float)


bucket_mean1=[]
bucket_std1 =[]
average_deviation=[]
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
    
    arr_diff.append(diff1[i])
    arr_diff.append(diff2[i])
    arr_diff.append(diff3[i])
    arr_diff=np.array(arr_diff,dtype=np.float)
    arr_diff_mean = np.nanmean(arr_diff)
    average_deviation.append(arr_diff_mean)
    
bucket_mean1[0]
bucket_std1[0]
diff1_std = np.nanstd(diff1)
diff2_std = np.nanstd(diff2)
diff3_std = np.nanstd(diff3)

sr=[]
for i in range(0, len(bucket1)):
    sr.append(i)



'''#Calculating diff12, diff23, diff31 and bucket_total( which is essentially MEAN of all matches the player has played) 
#there is no row in data where (bucket810=0 or blank) and (bucket1013=0 or blank) and (bucket1316=0 or blank)
#thats because there is no row
diff12= []
diff23=[]
diff31 =[]
bucket_total= []
for i in range(0,len(bucket1316)):
    d1=1
    d2=1
    d3=1
    a= bucket810[i]
    b= bucket1013[i]
    c= bucket1316[i]
    if (bucket810[i]==0 or bucket810[i] is None):
        d1=0  
        a=0
    
    if (bucket1013[i]==0 or bucket1013[i] is None):
        d2=0
        b=0
    if (bucket1316[i]==0 or bucket1316[i] is None):
        d3=0
        c=0
        
    num= a+b+c
    denom= d1+d2+d3
    bucket_total.append(num/denom)
    
    if ((bucket810[i]==0 or bucket810[i] is None) and (bucket1013[i]==0 or bucket1013[i] is None)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif ((bucket1013[i]==0 or bucket1013[i] is None) and (bucket1316[i]==0 or bucket1316[i] is None)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif ((bucket810[i]==0 or bucket810[i] is None) and (bucket1316[i]==0 or bucket1316[i] is None)):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(np.nan)
        
    elif (bucket810[i]==0 or bucket810[i] is None):
        diff12.append(np.nan)
        diff23.append(abs(bucket1013[i]-bucket1316[i]))
        diff31.append(np.nan)
    elif (bucket1013[i]==0 or bucket1013[i] is None):
        diff12.append(np.nan)
        diff23.append(np.nan)
        diff31.append(abs(bucket810[i]-bucket1316[i]))
    elif (bucket1316[i]==0 or bucket1316[i] is None):
        diff12.append(abs(bucket810[i]-bucket1013[i]))
        diff23.append(np.nan)
        diff31.append(np.nan)
    else:
        diff12.append(abs(bucket810[i]-bucket1013[i]))
        diff23.append(abs(bucket1013[i]-bucket1316[i]))
        diff31.append(abs(bucket810[i]-bucket1316[i])) '''
        



       
'''#calculating diff_mean
diff_mean=[]
for i in range(0,len(bucket1316)):
    d1=1
    d2=1
    d3=1
    a= diff12[i]
    b= diff23[i]
    c= diff31[i]
    if (diff12[i] is None):
        d1=0  
        a=0
    
    if (diff23[i] is None):
        d2=0
        b=0
    if (diff31[i] is None):
        d3=0
        c=0
        
    num= a+b+c
    denom= d1+d2+d3
    if(denom !=0):
        mean= num/denom
    else:
        mean=None
    diff_mean.append(mean)'''
      
#Adding bucket810, bucket1013, bucket1316, bucket_total, diff12, diff23, diff31, diff_mean to "Player_Attributes" 
df['Player_Attributes']['bucket810'] = bucket1
df['Player_Attributes']['bucket1013'] = bucket2
df['Player_Attributes']['bucket1316'] = bucket3
df['Player_Attributes']['bucket_total'] = bucket_total
df['Player_Attributes']['diff12'] = diff12
df['Player_Attributes']['diff23'] = diff23
df['Player_Attributes']['diff31'] = diff31
df['Player_Attributes']['diff_mean'] = diff_mean
df['Player_Attributes']['bucket_mean1'] = bucket_mean1
df['Player_Attributes']['bucket_std1'] = bucket_std1
df['Player_Attributes']['diff1'] = diff1
df['Player_Attributes']['diff2'] = diff2
df['Player_Attributes']['diff3'] = diff3
df['Player_Attributes']['average_deviation'] = average_deviation
df['Player_Attributes']['sr'] = sr
df['Player_Attributes'].to_csv('Player_Attributes_new1504_v2' + '.csv', index_label='index') 



r= np.unique(df['Player_Attributes'].player_api_id, return_index=True, return_inverse=True, return_counts=True)
#extracting 1 row for each player id into a separate data frame for analysis

r[0][1:10]
r[1][1:10]
indices =r[1]+1

unique_player_rows = df['Player_Attributes'][(df['Player_Attributes'].id.isin(indices))]
df_unique = pd.DataFrame(data= unique_player_rows)
df_unique.bucket810.describe()
#analysis
len(df_unique) 
#11060 unique players

df_unique.bucket_std1.describe()

#count    11060.000000
#mean         0.152850
#std          0.315145
#min          0.000000
#25%          0.000000
#50%          0.000000
#75%          0.173537
#max          3.500000


count_0= sum(df_unique.bucket_std1==0) # 6608
#percentage of NaN values in average_deviation

per_0 =count_0/len(df_unique)*100 #59.747%

#bucket_std1 =0 values imply the following:
# 1. Player has played in only one of the 3 time buckets
# 2. Playefr has played in exactly the same position in 2 or more time buckets

#There are 59.747% of such abover players

bucket_std_not0 = df_unique.bucket_std1[df_unique.bucket_std1>0]
len(bucket_std_not0) # 4452

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating histogram')
bp=ax.hist(bucket_std_not0, bins =14, color="purple")
print('saving figure')
fig.savefig('bucket_std1_hist.png')
plt.close(fig) 

#We notice maximum number of records to have standard deviation between 0 and 1
# standard deviation of records over 0.5 needs a more closer look

df_stdoverpoint5 = df_unique.bucket_mean1[df_unique.bucket_std1>0.5]

df_stdoverpoint5.describe()

#count    1133.000000
#mean        6.733434
#std         2.024886
#min         1.666667
#25%         4.845238
#50%         6.767034
#75%         8.532609
#max        10.444629

per_overpoint5= len(df_stdoverpoint5)/len(df_unique) *100 # 10.244% of the players have had over 0.5 units of standard deviation in their play position

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating boxplot')
bp=ax.boxplot(df_stdoverpoint5)
print('saving figure')
fig.savefig('bucket_mean1_stdoverpoint5.png')
plt.close(fig) 

# The above boxplot suggests that 50% of the players whose standard deviation in position is over 0.5, have bucket_mean1 from4.84 to 8.53
# This is not alarming


df_stdover1 = df_unique.bucket_mean1[df_unique.bucket_std1>1]

df_stdover1.describe()

#count    330.000000
#mean       6.196546
#std        1.740329
#min        3.807018
#25%        4.729620
#50%        5.537900
#75%        7.973095
#max        9.96739

per_over1= len(df_stdover1)/len(df_unique) *100 # 2.98% of the players have had over 0.5 units of standard deviation in their play position

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating boxplot')
bp=ax.boxplot(df_stdover1)
print('saving figure')
fig.savefig('bucket_mean1_stdover1.png')
plt.close(fig) 

# The above boxplot suggests that 50% of the players whose standard deviation in position is over 1, have bucket_mean1 from 4.72 to 7.97
# This is not alarming either

# identifying the borderline players 

def pos(x):
    if (x>=5 and x<10):
        return 3
    elif(x>1 and x<5):
        return 2
    elif(x >=10):
        return 4
    elif (np.isnan(x)):
        return "nan"
    elif(x ==1):
        return 1
    
    else:
        return 0
    
def get_pos(x):
    if (x==1):
        return "gk"
    elif(x==2):
        return "def"
    elif(x==3):
        return "mid"
    elif (x==4):
        return "Attacker"
      
    else:
        return 0
    
        
df_unique.bucket810.describe()
df_unique.bucket1013.describe()
df_unique.bucket1316.describe()
bucket1_np= np.array(df_unique.bucket810,dtype=float)
bucket2_np= np.array(df_unique.bucket1013,dtype=float)
bucket3_np= np.array(df_unique.bucket1316,dtype=float)
bucketmean_np =np.array(df_unique.bucket_mean1,dtype=float)
playerid_np =np.array(df_unique.player_api_id,dtype=int)
x= pos(bucket1_np[1])

np.isnan(bucket1_np[134])



position={}

for i in range(0,len(playerid_np)):
    result=[]
    result.append(pos(bucket1_np[i]))
    #print(bucket1_np[i])
    #print(pos(bucket1_np[i]))
    result.append(pos(bucket2_np[i]))
    #print(bucket2_np[i])
    #print(pos(bucket2_np[i]))
    result.append(pos(bucket3_np[i]))
    #print(bucket3_np[i])
    #print(pos(bucket3_np[i]))
    result.append(pos(bucketmean_np[i]))
    #print(bucketmean_np[i])
    #print(pos(bucketmean_np[i]))
    arr_result=np.array(result,dtype=float)
    #print(arr_result)
    y= np.nanstd(arr_result)
    #print(y)
    if (y==0):
      position[playerid_np[i]]=get_pos(pos(bucketmean_np[i]))
      
      #print(playerid_np[i])
      #print(get_pos(pos(bucketmean_np[i])))
             
            
    else:
        position[playerid_np[i]]="borderline"
        
players=[]
count=0
player_position=[]
for k in position:
    players.append(k)
    player_position.append(position[k])
    if (position[k]== "borderline"):
        count=count+1
    
count  #814 players are on "borderline"... as in played at multiple positions  
count/len(df_unique)*100 #7.36% of my players have played in more than 1 position over time.

#Since this is a significant number, we will divide our positions into 6 buckets instead of 4 buckets. 
# But first, lets find out the exact variation of positions in these identified "borderline" players


