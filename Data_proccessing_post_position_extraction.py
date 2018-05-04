# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 19:04:16 2018

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
#------------------------
table = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table'", conn)
df={}
for n in table['name']:
    print(n)
    
    df[n]=pd.read_sql_query("SELECT * from %s" % n, conn)
    df[n].to_csv(n + '.csv', index_label='index')
    
#locals().update(df) ------->>> is messing with local namespace a bad idea?? research needed
print(df.keys())

#Adding extracted position to "Player_Attributes"

df['Player_Attributes'].shape
p= pd.read_csv('F:/Player-Analytics/kmean_predicted_on_bucket_mean.csv')
list(p)
p1=p.drop('bucket_mean1',1)
p1=p1.drop('bucket_std1',1)
p1=p1.drop('kmean_predict',1)
p1=p1.drop('Unnamed: 0',1)
p1=p1.drop( 'index',1)
p1['position']=p1['kmeans_predict_position']
p1.position
p1=p1.drop('kmeans_predict_position',1)
list(p1)

r= np.unique(p.player_api_id, return_index=True, return_inverse=True, return_counts=True)

df['Player_Attributes'] =p1
df['Player_Attributes'].shape
df['Player_Attributes'].to_csv('Player_Attributes_new' + '.csv', index_label='index')   

nanrows=df['Player_Attributes'][df['Player_Attributes'].isnull().T.any().T]
nanrows.shape # 0.02%
df['Player_Attributes']=df['Player_Attributes'].dropna()
df['Player_Attributes'].shape
df['Player_Attributes'].to_csv('Player_Attributes_new1' + '.csv', index_label='index')   

r1= np.unique(df['Player_Attributes'].player_api_id, return_index=True, return_inverse=True, return_counts=True)

len(r1[0])

#180354 rows (prev 183978)
#10410 players (prev 11060)

#Merge with additonal player data like birthdate,etc
player_data=pd.merge(df['Player_Attributes'], df['Player'], how='inner', on=['player_api_id'])


#Dropping duplicate columns
player_data=player_data.loc[:, player_data.columns!='Unnamed: 0_x']
player_data=player_data.loc[:, player_data.columns!='id_x']
player_data=player_data.loc[:, player_data.columns!='Unnamed: 0_y']
player_data=player_data.loc[:, player_data.columns!='id_y']
player_data=player_data.loc[:, player_data.columns!='player_fifa_api_id_y']
#Change of column name
player_data.rename(columns={'player_fifa_api_id_x': 'player_fifa_api_id'}, inplace=True)



nanrows=player_data[player_data.isnull().T.any().T]
nanrows.shape
#0 NA rows
player_data.shape

player_data['attacking_work_rate'].unique()
player_data['defensive_work_rate'].unique()
player_data['preferred_foot'].unique()

"""
Multiple instances of work rate and preferred foot being labelled incorrectly.
This method would drop all the rows containing uncertain values and map the 
data to correct key as seen below
""" 
def prep_player_data (df):
    work_rate_dict = {'low': 'low', 'medium': 'medium', 'high': 'high'}
    pref_foot_dict = {'left': 'left', 'right': 'right', 'None': 'None'}

    df = df.loc[(df['attacking_work_rate'].isin(work_rate_dict.keys())) & 
                (df['defensive_work_rate'].isin(work_rate_dict.keys()))].copy()
    
    df.loc[:, 'preferred_foot'] = df.loc[:, 'preferred_foot'].map(pref_foot_dict)
    df.loc[:, 'attacking_work_rate'] = df.loc[:, 'attacking_work_rate'].map(work_rate_dict)
    df.loc[:, 'defensive_work_rate'] = df.loc[:, 'defensive_work_rate'].map(work_rate_dict)
    
    return df

player_data=prep_player_data(player_data) #176161 rows
player_data.shape
print(player_data.head(1))
player_data['player_api_id'].astype('category')

def get_age(x1,x2):
    bday  =  x1.split(" ")[0]
    fifa  =  x2.split(" ")[0]
    bday = datetime.datetime.strptime(bday, "%Y-%m-%d").date()
    fifa = datetime.datetime.strptime(fifa, "%Y-%m-%d").date()
    return fifa.year - bday.year - ((fifa.month, fifa.day) < (bday.month, bday.day))

player_data["age"] = np.vectorize(get_age)(player_data["birthday"],player_data['date'])
player_data.shape
print(player_data.head(1))

player_data['player_fifa_api_id']=player_data['player_fifa_api_id'].astype(np.int64)
player_data['player_api_id']=player_data['player_api_id'].astype(np.int64)
player_data['date']=player_data['date'].astype(np.datetime64) ##Future warning??
player_data['overall_rating']=player_data['overall_rating'].astype(np.int16)
player_data['potential']=player_data['potential'].astype(np.int16)
player_data['preferred_foot']=player_data['preferred_foot'].astype('category')
player_data['attacking_work_rate']=player_data['attacking_work_rate'].astype('category')
player_data['defensive_work_rate']=player_data['defensive_work_rate'].astype('category')
player_data['crossing']=player_data['crossing'].astype(np.int16)
player_data['finishing']=player_data['finishing'].astype(np.int16)
player_data['heading_accuracy']=player_data['heading_accuracy'].astype(np.int16)
player_data['short_passing']=player_data['short_passing'].astype(np.int16)
player_data['volleys']=player_data['volleys'].astype(np.int16)
player_data['dribbling']=player_data['dribbling'].astype(np.int16)
player_data['curve']=player_data['curve'].astype(np.int16)
player_data['free_kick_accuracy']=player_data['free_kick_accuracy'].astype(np.int16)
player_data['long_passing']=player_data['long_passing'].astype(np.int16)
player_data['ball_control']=player_data['ball_control'].astype(np.int16)
player_data['acceleration']=player_data['acceleration'].astype(np.int16)
player_data['sprint_speed']=player_data['sprint_speed'].astype(np.int16)
player_data['agility']=player_data['agility'].astype(np.int16)
player_data['reactions']=player_data['reactions'].astype(np.int16)
player_data['balance']=player_data['balance'].astype(np.int16)
player_data['shot_power']=player_data['shot_power'].astype(np.int16)
player_data['jumping']=player_data['jumping'].astype(np.int16)
player_data['stamina']=player_data['stamina'].astype(np.int16)
player_data['strength']=player_data['strength'].astype(np.int16)
player_data['long_shots']=player_data['long_shots'].astype(np.int16)
player_data['aggression']=player_data['aggression'].astype(np.int16)
player_data['interceptions']=player_data['interceptions'].astype(np.int16)
player_data['positioning']=player_data['positioning'].astype(np.int16)
player_data['vision']=player_data['vision'].astype(np.int16)
player_data['penalties']=player_data['penalties'].astype(np.int16)
player_data['marking']=player_data['marking'].astype(np.int16)
player_data['standing_tackle']=player_data['standing_tackle'].astype(np.int16)
player_data['sliding_tackle']=player_data['sliding_tackle'].astype(np.int16)
player_data['gk_diving']=player_data['gk_diving'].astype(np.int16)
player_data['gk_handling']=player_data['gk_handling'].astype(np.int16)
player_data['gk_kicking']=player_data['gk_kicking'].astype(np.int16)
player_data['gk_positioning']=player_data['gk_positioning'].astype(np.int16)
player_data['gk_reflexes']=player_data['gk_reflexes'].astype(np.int16)
player_data['player_name']=player_data['player_name'].astype(np.str)
player_data['birthday']=player_data['birthday'].astype(np.str)
player_data['height']=player_data['height'].astype(np.int16)
player_data['weight']=player_data['weight'].astype(np.int16)
player_data['age']=player_data['age'].astype(np.int16)
player_data['position']=player_data['position'].astype('category')

#Dropping duplicate rows
player_data=player_data.drop_duplicates()

player_data['attacking_work_rate'].values
len(player_data)

for col in player_data.columns:
    print(col)
    if (player_data[col].dtypes == 'int16' or player_data[col].dtypes == 'int64'):
        print('creating fig')
        fig=plt.figure()
        print('creating subplot')
        ax=fig.add_subplot(111)
        print('creating boxplot')
        bp=ax.boxplot(player_data[col])
        print('saving figure')
        fig.savefig(col+'.png')
        plt.close(fig)
        
    if (player_data[col].dtype.name == 'category'):
        print('creating fig')
        plot=player_data[col].value_counts().plot(kind='bar')
        fig=plot.get_figure()
        print('creating subplot')
        print('creating barchart')     
        print('saving figure')
        fig.savefig(col+'.png')
        plt.close(fig)
        
from sklearn.model_selection import train_test_split
X=player_data
y=player_data.overall_rating
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

X_train.shape
X_train.boxplot('potential')
X_train.to_csv('F:/Player-Analytics/training set_new1.csv')
X_test
X_test.boxplot('potential')
X_test.to_csv('F:/Player-Analytics/test set_new1.csv')
player_data.to_csv('player_data.csv')
