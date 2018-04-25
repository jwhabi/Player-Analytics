# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 03:34:56 2018

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

df['Player_Attributes'] = pd.read_csv('F:/Player-Analytics/Player_Attributes_v22.csv')
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
    elif(x <=1):
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
position1={}
position2={}
position3={}

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
      position[playerid_np[i]]=pos(bucketmean_np[i])
      
      #print(playerid_np[i])
      #print(get_pos(pos(bucketmean_np[i])))
             
            
    else:
        position[playerid_np[i]]=5
        position1[playerid_np[i]]= pos(bucket1_np[i]) 
        position2[playerid_np[i]]= pos(bucket2_np[i])
        position3[playerid_np[i]]= pos(bucket3_np[i])
        
        
len(position1)# 807
len(position2)#807
len(position3)#807  
        
   

new_stuff=pd.DataFrame.from_dict(position,orient='index')
new_stuff=new_stuff.reset_index()
new_stuff.columns=['player_api_id','position']

new_stuff.to_csv('position_unique' + '.csv', index_label='index') 
new_stuff.position.describe()
new_stuff.position.value_counts()

#3    4734 - 42.80%
#2    3431 - 31.02%
#4    1147 - 10.37%
#1     941 - 8.5%
#5     807 - 7.3%

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating barplot')


plot=new_stuff.position.value_counts().plot(kind='bar')
fig=plot.get_figure()

print('saving figure')
fig.savefig('player_position_distribution.png')
plt.close(fig) 

#814 players are on "borderline"... as in played at multiple positions  
#7.36% of my players have played in more than 1 position over time.

#Since this is a significant number, we will divide our positions into 6 buckets instead of 4 buckets. 
# But first, lets find out the exact variation of positions in these identified "borderline" players

    
PA_all = pd.DataFrame(df['Player_Attributes'])
PA_all1=PA_all.drop('Position',1)
PA_all2=PA_all1.drop('average_deviation',1)
PA_all3=PA_all2.drop('average deviation',1)
PA_all4=PA_all3.drop('diff1',1)
PA_all5=PA_all4.drop('diff2',1)
PA_all6=PA_all5.drop('diff3',1)
PA_all7=PA_all6.drop('diff_mean',1)
PA_all8=PA_all7.drop('diff12',1)
PA_all9=PA_all8.drop('diff23',1)
PA_all10=PA_all9.drop('diff31',1)

PA = PA_all10


new_PA=pd.merge(PA, new_stuff, how='left', on=['player_api_id'])


fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating barplot')
plot=new_PA.position.value_counts().plot(kind='bar')
fig=plot.get_figure()
print('saving figure')
fig.savefig('player_position_distribution_new_PA.png')
plt.close(fig) 

len(new_PA)
new_PA.position.value_counts()

#3    81244 - 44.16%
#2    53504 - 29.08%
#4    18133 - 9.8%
#5    16441 - 8.9% - significant
#1    14656 - 7.9%


new_PA['position1'] =np.where(new_PA['position']!= 5, new_PA['position'],new_PA['year'])
#new_PA.to_csv('new_PA' + '.csv', index_label='index') 

new_PA['position2'] =np.where((new_PA['position1']== 2008) | (new_PA['position1']==2009) | (new_PA['position1']== 2010), new_PA['bucket810'],new_PA['position1'])
new_PA['position2'] =np.where((new_PA['position1']== 2011) | (new_PA['position1']==2012) | (new_PA['position1']== 2013), new_PA['bucket1013'],new_PA['position2'])
new_PA['position2'] =np.where((new_PA['position1']== 2014) | (new_PA['position1']==2015) | (new_PA['position1']== 2016), new_PA['bucket1316'],new_PA['position2'])
new_PA['position2'] =np.where(new_PA['position1']== 2007, new_PA['bucket_mean1'],new_PA['position2'])
new_PA['position2'] =np.where(new_PA['position2'].isnull(), new_PA['bucket_mean1'],new_PA['position2'])

new_PA['position2']=new_PA['position2'].astype(np.float32)
new_PA.position2.describe()

new_PA['position3'] =np.where((new_PA['position']==5)& (new_PA['position2']>=10), 40 ,new_PA['position2'])
new_PA['position3'] =np.where((new_PA['position']==5)& (new_PA['position3']<10), 30 ,new_PA['position3'])
new_PA['position3'] =np.where((new_PA['position']==5)& (new_PA['position2']<5), 20 ,new_PA['position3'])
new_PA['position3'] =np.where((new_PA['position']==5)& (new_PA['position2']<2), 10 ,new_PA['position3'])

new_PA['position3'] =np.where((new_PA['position3']==10), 1 ,new_PA['position3'])
new_PA['position3'] =np.where((new_PA['position3']==20), 2,new_PA['position3'])
new_PA['position3'] =np.where((new_PA['position3']==30), 3 ,new_PA['position3'])
new_PA['position3'] =np.where((new_PA['position3']==40), 4 ,new_PA['position3'])

new_PA.to_csv('PA_afterQA' + '.csv', index_label='index') 





