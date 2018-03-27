# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 21:02:44 2018

@author: Vaibhav
"""

import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import matplotlib.pyplot as plt
import datetime
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

#Setting path
project_path='C:\Masters\Data Preparation and Analysis\Project'
os.chdir(project_path)
#os.getcwd()

#Reding Player.csv
player=pd.read_csv('Player.csv')
#player.head()

#Reading Player_Attribute.csv
player_attrib=pd.read_csv('Player_Attributes.csv')
#player_attrib.head()

#Performing inner join
player_data=pd.merge(player_attrib, player, how='inner', on=['player_api_id'])
#player_data.head()

#Dropping duplicate columns
player_data=player_data.loc[:, player_data.columns!='Unnamed: 0_x']
player_data=player_data.loc[:, player_data.columns!='id_x']
player_data=player_data.loc[:, player_data.columns!='Unnamed: 0_y']
player_data=player_data.loc[:, player_data.columns!='id_y']
player_data=player_data.loc[:, player_data.columns!='player_fifa_api_id_y']
#Change of column name
player_data.rename(columns={'player_fifa_api_id_x': 'player_fifa_api_id'}, inplace=True)
#player_data.head()
#player_data.describe()#183978

#list of observations with NaN values for any variable
nanrows=player_data[player_data.isnull().T.any().T]
nanrows.shape#3624
#nanrows=player_data[player_data['overall_rating'].isnull()]

"""3624 is nearly 2% of the player data available. Hence dropping this would not 
affect our analysis"""
player_data=player_data.dropna()#180354

###############################################################################
#Saving to disk
#player_data.to_csv('Player_Data.csv')
###############################################################################

#player_data.columns
"""
['player_fifa_api_id', 'player_api_id', 'date', 'overall_rating',
       'potential', 'preferred_foot', 'attacking_work_rate',
       'defensive_work_rate', 'crossing', 'finishing', 'heading_accuracy',
       'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy',
       'long_passing', 'ball_control', 'acceleration', 'sprint_speed',
       'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
       'strength', 'long_shots', 'aggression', 'interceptions', 'positioning',
       'vision', 'penalties', 'marking', 'standing_tackle', 'sliding_tackle',
       'gk_diving', 'gk_handling', 'gk_kicking', 'gk_positioning',
       'gk_reflexes', 'player_name', 'birthday', 'height', 'weight'],
      dtype='object'
"""

player_attrib['attacking_work_rate'].astype('category')
player_attrib['preferred_foot'].values

player_attrib.loc[927,:]
player_data.loc[1:10,:]
player_data.sort_index().loc[920:930]
player_data.shape

"""
Multiple instances of work rate and preferred foot being labelled incorrectly.
This method would drop all the rows containing uncertain values and map the 
data to correct key as seen below
""" 
def prep_player_data (df):
    work_rate_dict = {'low': 0, 'medium': 1, 'high': 2}
    pref_foot_dict = {'left': 0, 'right': 1, 'None': 2}

    df = df.loc[(df['attacking_work_rate'].isin(work_rate_dict.keys())) & 
                (df['defensive_work_rate'].isin(work_rate_dict.keys()))].copy()
    
    df.loc[:, 'preferred_foot'] = df.loc[:, 'preferred_foot'].map(pref_foot_dict)
    df.loc[:, 'attacking_work_rate'] = df.loc[:, 'attacking_work_rate'].map(work_rate_dict)
    df.loc[:, 'defensive_work_rate'] = df.loc[:, 'defensive_work_rate'].map(work_rate_dict)
    
    return df

player_data=prep_player_data(player_data) #176161 rows
player_data.head()
player_data['player_api_id'].astype('category')

def get_age(x1,x2):
    bday  =  x1.split(" ")[0]
    fifa  =  x2.split(" ")[0]
    bday = datetime.datetime.strptime(bday, "%Y-%m-%d").date()
    fifa = datetime.datetime.strptime(fifa, "%Y-%m-%d").date()
    return fifa.year - bday.year - ((fifa.month, fifa.day) < (bday.month, bday.day))

player_data["age"] = np.vectorize(get_age)(player_data["birthday"],player_data['date'])

player_data['player_fifa_api_id']=player_data['player_fifa_api_id'].astype(np.int64)
player_data['player_api_id']=player_data['player_api_id'].astype(np.int64)
player_data['date']=player_data['date'].astype(np.str)
player_data['overall_rating']=player_data['overall_rating'].astype(np.int16)
player_data['potential']=player_data['potential'].astype(np.int16)
player_data['preferred_foot']=player_data['preferred_foot'].astype(np.int16)
player_data['attacking_work_rate']=player_data['attacking_work_rate'].astype(np.int16)
player_data['defensive_work_rate']=player_data['defensive_work_rate'].astype(np.int16)
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

os.chdir(project_path+'\plots')
for col in player_data.columns:
    if player_data[col].dtypes != 'O':
        print('creating fig')
        fig=plt.figure()
        print('creating subplot')
        ax=fig.add_subplot(111)
        print('creating boxplot')
        bp=ax.boxplot(player_data[col])
        print('saving figure')
        fig.savefig(col+'.png')
        plt.close(fig)
os.chdir(project_path)
#Generating histograms
os.chdir(project_path+'\plots')
for col in player_data.columns:
    if player_data[col].dtypes != 'O':
        print('creating fig')
        fig=plt.figure()
        print('creating subplot')
        ax=fig.add_subplot(111)
        print('creating boxplot')
        bp=ax.hist(player_data[col])
        print('saving figure')
        fig.savefig(col+'_hist.png')
        plt.close(fig)
os.chdir(project_path)


from sklearn.model_selection import train_test_split
X=player_data
y=player_data.overall_rating
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

X_train.shape
X_train.boxplot('potential')
X_train.to_csv('training set.csv')
X_test
X_test.boxplot('potential')
X_test.to_csv('test set.csv')
