# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 14:07:10 2018

@author: jaide
"""

import sqlite3 
import pandas as pd
import numpy as np
import sklearn
import os
import matplotlib.pyplot as plt
import datetime
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')


X_train=pd.read_csv('C:\\Users\\jaide\\Downloads\\training set_new1.csv')
X_test=pd.read_csv('C:\\Users\\jaide\\Downloads\\test set_new1.csv')
pos_dict = {'for': 3, 'mid': 2, 'def': 1, 'gk': 0}
X_train.loc[:, 'position'] = X_train.loc[:, 'position'].map(pos_dict)
X_test.loc[:, 'position'] = X_test.loc[:, 'position'].map(pos_dict)
y_train=X_train.position
y_test=X_test.position

X_train=X_train.drop('Unnamed: 0',axis=1)
X_test=X_test.drop('Unnamed: 0',axis=1)

X_train=X_train.drop('position',axis=1)
X_test=X_test.drop('position',axis=1)
X_test.shape
X_train.shape
len(y_train)

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
X_test=X_test.drop('potential',1)
X_train=X_train.drop('potential',1)


X_test=X_test.drop('defensive_work_rate',1)
X_train=X_train.drop('defensive_work_rate',1)
X_test=X_test.drop('attacking_work_rate',1)
X_train=X_train.drop('attacking_work_rate',1)
X_test=X_test.drop('preferred_foot',1)
X_train=X_train.drop('preferred_foot',1)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model= gnb.fit(X_train, y_train)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred.dtype
model.score(X_test,y_test)
print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0],(y_test != y_pred).sum()))
from Eval import Eval
eval1 = Eval(y_pred, np.array(y_test))

print("Positive Class:")
print("Accuracy: ",eval1.Accuracy())
from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))

import pickle
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.score(X_test,y_test)
#X_test.to_csv('sample_players1.csv')
#y_test.to_csv('pred.csv')
#X_test.iloc[:,3:39].head(1)
#loaded_model.theta_.shape
#X_test.shape
#y_test=pd.read_csv('C:\\Users\\jaide\\pred.csv',header=None)
#X_test=pd.read_csv('C:\\Users\\jaide\\new_file.csv')
#X_test=X_test.drop('Unnamed: 0',axis=1)
#y_test.columns
#len(y_test.reset_index(drop=True))
#len(y_test)
#loaded_model.score(X_test,y_test)
#sample=pd.read_csv('C:\\Users\\jaide\\sample_players.csv')
#sample.iloc[:,3:42]
#loaded_model.predict(sample.iloc[:,3:42])

#cols=['gk_diving', 'finishing','standing_tackle','sliding_tackle','interceptions','marking','volleys','weight','strength','jumping']

#loaded_model.predict(sample.filter(items=cols))
        
