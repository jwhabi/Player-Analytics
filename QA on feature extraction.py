# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 03:34:56 2018

@author: pragy
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


a = pd.read_csv('F:/Player-Analytics/Player_Attributes_v22.csv')
r= np.unique(a.player_api_id, return_index=True, return_inverse=True, return_counts=True)
#extracting 1 row for each player id into a separate data frame for analysis

r[0][1:10]
r[1][1:10]
indices =r[1]+1

unique_player_rows = a[(a.id.isin(indices))]
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
# 2. Player has played in exactly the same position in 2 or more time buckets

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

# The above boxplot suggests that 50% of the players whose standard deviation in position is over 0.5, have bucket_mean1 from 4.84 to 8.53
# This is not alarming.


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

per_over1= len(df_stdover1)/len(df_unique) *100 # 2.98% of the players have had over 1 units of standard deviation in their play position

fig=plt.figure()
print('creating subplot')
ax=fig.add_subplot(111)
print('creating boxplot')
bp=ax.boxplot(df_stdover1)
print('saving figure')
fig.savefig('bucket_mean1_stdover1.png')
plt.close(fig) 

# The above boxplot suggests that 50% of the players whose standard deviation in position is over 1, have bucket_mean1 from 4.72 to 7.97
# This is not alarming either.
