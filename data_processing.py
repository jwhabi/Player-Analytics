# -*- coding: utf-8 -*-
"""
Created on Wed Apr  4 22:34:45 2018

@author: jaide
"""

import sqlite3 
import pandas as pd
import numpy as np

import os
import matplotlib.pyplot as plt
import datetime
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

conn = sqlite3.connect("C:\\Users\\jaide\\Downloads\\soccer\\database.sqlite")

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

#Evaluate the labelled position for each player based on match lineups
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
        
#Test: (uncomment any below lines to test the function)
#get_position(36835)    
#get_position(38788)
#get_position(94462)
#get_position(37069)
#get_position(50160)
pos=[]


for i in range(0,len(df['Player_Attributes'])):
    pos.append(get_position(df['Player_Attributes'].player_api_id[i]))
    
df['Player_Attributes']['position'] = pos
df['Player_Attributes'].to_csv('Player_Attributes' + '.csv', index_label='index')   

nanrows=df['Player_Attributes'][df['Player_Attributes'].isnull().T.any().T]
nanrows.shape

df['Player_Attributes']=df['Player_Attributes'].dropna()
df['Player_Attributes'].shape
df['Player_Attributes'].to_csv('Player_Attributes' + '.csv', index_label='index')   

#180195 rows (prev 183978)
#10403 players (prev 11060)

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
    #work_rate_dict = {'low': 'low', 'medium': 'medium', 'high': 'high'}
    #pref_foot_dict = {'left': 'left', 'right': 'right', 'None': 'None'}
    work_rate_dict = {'low': 0, 'medium': 1, 'high': 2}
    pref_foot_dict = {'left': 1, 'right': 2, 'None': 0}
    pos_dict = {'for': 3, 'mid': 2, 'def': 1, 'gk': 0}

    df = df.loc[(df['attacking_work_rate'].isin(work_rate_dict.keys())) & 
                (df['defensive_work_rate'].isin(work_rate_dict.keys()))].copy()
    
    df.loc[:, 'preferred_foot'] = df.loc[:, 'preferred_foot'].map(pref_foot_dict)
    df.loc[:, 'position'] = df.loc[:, 'position'].map(pos_dict)
    df.loc[:, 'attacking_work_rate'] = df.loc[:, 'attacking_work_rate'].map(work_rate_dict)
    df.loc[:, 'defensive_work_rate'] = df.loc[:, 'defensive_work_rate'].map(work_rate_dict)
    
    return df

player_data=prep_player_data(player_data) #176002 rows
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
player_data['date']=player_data['date'].astype(np.str) ##Future warning??
player_data['overall_rating']=player_data['overall_rating'].astype(np.int16)
player_data['potential']=player_data['potential'].astype(np.int16)

#player_data['preferred_foot']=player_data['preferred_foot'].astype('category')
#player_data['attacking_work_rate']=player_data['attacking_work_rate'].astype('category')
#player_data['defensive_work_rate']=player_data['defensive_work_rate'].astype('category')
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
player_data['position']=player_data['position'].astype('category')


player_data['attacking_work_rate'].values
player_data.shape

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
        
player_data.describe(include= 'all')
player_data.dtypes
      
from sklearn.model_selection import train_test_split
X=player_data
X=X.drop(['position'],axis=1)
y=player_data.position
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

X_train.shape
X_train.boxplot('potential')
X_train.to_csv('training set.csv')
X_test.shape
X_test.boxplot('potential')
X_test.to_csv('test set.csv')

X_train=X_train.drop('date',1)
X_test=X_test.drop('date',1)
X_train=X_train.drop('birthday',1)
X_test=X_test.drop('birthday',1)
X_train=X_train.drop('player_name',1)
X_test=X_test.drop('player_name',1)
X_test=X_test.drop('player_api_id',1)
X_train=X_train.drop('player_api_id',1)
X_test=X_test.drop('player_fifa_api_id',1)
X_train=X_train.drop('player_fifa_api_id',1)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred.dtype
print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0],(y_test != y_pred).sum()))
from Eval import Eval
eval1 = Eval(y_pred, np.array(y_test))

print("Positive Class:")
print("Accuracy: ",eval1.Accuracy())
from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))
print("Recall: ",eval1.Recall())
print("Precision: ",eval1.Precision())

means=pd.DataFrame(columns=['Feature','diff_abs_mean','diff_var'])
diff=[]
ad=[]
for i in range(0,gnb.theta_.shape[1]):
    diff.append(abs(gnb.theta_[0,i]-gnb.theta_[1,i]))
    ad.append(gnb.sigma_[0,i]+gnb.sigma_[1,i])
    means=means.append([{'Feature': X_train.columns[i],'diff_abs_mean': abs(gnb.theta_[0,i]-gnb.theta_[1,i]), 'diff_var': gnb.sigma_[0,i]+gnb.sigma_[1,i]}])
means['measure']=means.diff_abs_mean/means.diff_var
means=means.reset_index()
means=means.drop(['index','diff_abs_mean','diff_var'],1)
means.sort_values(by=['measure'], ascending = False)


from sklearn.model_selection import cross_val_score
cross_fold = pd.DataFrame(columns=['Precision', 'Recall', 'Accuracy'])
cross_fold.Precision=cross_val_score(gnb,X_train,y_train,cv=10,scoring='precision_micro')

cross_fold.Recall=cross_val_score(gnb,X_train,y_train,cv=10,scoring='recall_micro')
cross_fold.Accuracy=cross_val_score(gnb,X_train,y_train,cv=10)
cross_fold

#--------------------------------------------------------


#def prep_player_data_reverse (df):
#    #work_rate_dict = {'low': 'low', 'medium': 'medium', 'high': 'high'}
#    #pref_foot_dict = {'left': 'left', 'right': 'right', 'None': 'None'}
#    work_rate_dict = {0: 'low', 1: 'medium', 2: 'high'}
#    pref_foot_dict = { 1:'left', 2:'right',  0: 'None'}
#    
#
#    df = df.loc[(df['attacking_work_rate'].isin(work_rate_dict.keys())) & 
#                (df['defensive_work_rate'].isin(work_rate_dict.keys()))].copy()
#    
#    df.loc[:, 'preferred_foot'] = df.loc[:, 'preferred_foot'].map(pref_foot_dict)
#    
#    df.loc[:, 'attacking_work_rate'] = df.loc[:, 'attacking_work_rate'].map(work_rate_dict)
#    df.loc[:, 'defensive_work_rate'] = df.loc[:, 'defensive_work_rate'].map(work_rate_dict)
#    df['attacking_work_rate']=df['attacking_work_rate'].astype('category')
#    df['defensive_work_rate']=df['defensive_work_rate'].astype('category')
#    df['preferred_foot']=df['preferred_foot'].astype('category')
#    return df
#
#X_train=prep_player_data_reverse(X_train)
#X_test=prep_player_data_reverse(X_test)
#X_train.dtypes
#X_test.dtypes
#pos_dict = { 3: 'for',  2: 'mid', 1: 'def', 0: 'gk'}
#df.loc[:, 'position'] = df.loc[:, 'position'].map(pos_dict)


def convert_to_categories(x):
    for i in x.columns:
        if x[i].dtype=='int16':
            category=pd.cut(np.array(x[i]), 3,labels=[0,1,2])
            x[i]=category
    
    return x

X_train=convert_to_categories(X_train)
X_train.dtypes
X_train.aggression


X_test=convert_to_categories(X_test)
X_test.dtypes
X_test.aggression

#y_train=y_train.map({ 3: 'for',  2: 'mid', 1: 'def', 0: 'gk'})
#y_test=y_test.map({ 3: 'for',  2: 'mid', 1: 'def', 0: 'gk'})
#
#y_train=y_train.astype('category')
#y_test=y_test.astype('category')


from sklearn.naive_bayes import MultinomialNB
gnb = MultinomialNB()
y_pred = gnb.fit(X_train, y_train).predict(X_test)
gnb.fit(X_train, y_train).predict(X_test.head(5))
print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0],(y_test != y_pred).sum()))
from Eval import Eval
eval1 = Eval(y_pred, np.array(y_test))

print("Positive Class:")
print("Accuracy: ",eval1.Accuracy())
from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))
print("Recall: ",eval1.Recall())
print("Precision: ",eval1.Precision())