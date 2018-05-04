
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

#Read the train and test csv files
X_train=pd.read_csv('C:\\Users\\jaide\\Downloads\\training set_new1.csv')
X_test=pd.read_csv('C:\\Users\\jaide\\Downloads\\test set_new1.csv')

#One-hot-encoding for Player data - categorical variables
def prep_player_data (df):
    
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

X_train=prep_player_data(X_train) 
X_train.shape #(140928, 49)
X_train.head(1)

X_test=prep_player_data(X_test) 
X_test.shape #(35233, 49)
X_test.head(1)


#Position is the variable to be predicted

y_train=X_train.position
y_test=X_test.position

#Drop unused features and/or features not useful like player id,etc
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
#Potential cannot be used since it is a final prediction in our other models
X_test=X_test.drop('potential',1)
X_train=X_train.drop('potential',1)


#Running a Gaussian Naive bayes model for prediction positions
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
model= gnb.fit(X_train, y_train)
y_pred = gnb.fit(X_train, y_train).predict(X_test)
y_pred.dtype
model.score(X_test,y_test)
print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0],(y_test != y_pred).sum()))
#Number of mislabeled points out of a total 35233 points : 7598
from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))
#Recall: [ 0.9971831   0.8077894   0.65244734  0.89250895]
#Precision: [ 0.99543058  0.81573197  0.74884773  0.73013629]
#Accuracy: 0.784349899242

#Getting the top significant features contributing to this model
#This is done by gathering the normalized means of the model features and sorting
#from high to low, the highest being the most significant
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

#Cross-fold validation for the model gnb
from sklearn.model_selection import cross_val_score
cross_fold = pd.DataFrame(columns=['Precision', 'Recall', 'Accuracy'])
cross_fold.Precision=cross_val_score(gnb,X_train,y_train,cv=10,scoring='precision_micro')

cross_fold.Recall=cross_val_score(gnb,X_train,y_train,cv=10,scoring='recall_micro')
cross_fold.Accuracy=cross_val_score(gnb,X_train,y_train,cv=10)
cross_fold
#Generates consistent results reflecting the scores already obtained

#Since Gaussian NB prefers pure gaussian features,
#let's try dropping the 3 categorical variables and see if it makes a difference
X_test=X_test.drop('defensive_work_rate',1)
X_train=X_train.drop('defensive_work_rate',1)
X_test=X_test.drop('attacking_work_rate',1)
X_train=X_train.drop('attacking_work_rate',1)
X_test=X_test.drop('preferred_foot',1)
X_train=X_train.drop('preferred_foot',1)


from sklearn.naive_bayes import GaussianNB
gnb2 = GaussianNB()
model= gnb2.fit(X_train, y_train)
y_pred = gnb2.fit(X_train, y_train).predict(X_test)
y_pred.dtype
model.score(X_test,y_test)
print("Number of mislabeled points out of a total %d points : %d"  % (X_test.shape[0],(y_test != y_pred).sum()))
#Number of mislabeled points out of a total 35233 points : 7618
from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))
#Recall: [ 0.9971831   0.8059863   0.65275042  0.8920153 ]
#Precision: [ 0.99543058  0.81480131  0.74763516  0.73061761]
#Accuracy: 0.783782249596

#Cross-fold validation for the model gnb2
from sklearn.model_selection import cross_val_score
cross_fold = pd.DataFrame(columns=['Precision', 'Recall', 'Accuracy'])
cross_fold.Precision=cross_val_score(gnb2,X_train,y_train,cv=10,scoring='precision_micro')

cross_fold.Recall=cross_val_score(gnb2,X_train,y_train,cv=10,scoring='recall_micro')
cross_fold.Accuracy=cross_val_score(gnb2,X_train,y_train,cv=10)
cross_fold
cross_fold.Accuracy.mean()
#Generates consistent results reflecting the scores already obtained

#Removing the categorical variables seems to not affect model accuracy too much,
#since the information lost is covered by the other variables.
#Thus is would be wiser to drop these 3 variables for our prediction instead of 
#having the risk of misappropriation by incorrect one-hot-encoding.


###########Trial###########
#Let's give Multinomial NB a try by converting all continous features to categorical
def convert_to_categories(x):
    for i in x.columns:
        if x[i].dtype=='int16':
            category=pd.cut(np.array(x[i]), 3,labels=[0,1,2])
            x[i]=category
    
    return x

X_train1=convert_to_categories(X_train)
X_train1.dtypes
X_train1.aggression


X_test1=convert_to_categories(X_test)
X_test1.dtypes
X_test1.aggression

#y_train=y_train.map({ 3: 'for',  2: 'mid', 1: 'def', 0: 'gk'})
#y_test=y_test.map({ 3: 'for',  2: 'mid', 1: 'def', 0: 'gk'})
#
#y_train=y_train.astype('category')
#y_test=y_test.astype('category')


from sklearn.naive_bayes import MultinomialNB
gnb3 = MultinomialNB()
y_pred = gnb3.fit(X_train1, y_train).predict(X_test1)
#gnb.fit(X_train, y_train).predict(X_test.head(5))
print("Number of mislabeled points out of a total %d points : %d"  % (X_test1.shape[0],(y_test != y_pred).sum()))

from sklearn.metrics import recall_score,precision_score,accuracy_score
print(recall_score(y_test,y_pred,average=None))
print(precision_score(y_test,y_pred,average=None))
print(accuracy_score(y_test,y_pred))
#Accuracy comes to 77% with significant reduction in Recall and Precision for mid and for categories.
#To perform a good multinomial fit a heavily modified version of this dataset will be required.
#Encoding continous variables to categorical results in a heavy loss of information

############Trial###############

#The model gnb2 is the chosen one since it has consistent Precision, Recall and Accuracy
#even after a 10-fold validation and has pure gaussian features and thus does not misappropriate data

#Creating a model object file on the system for use in implementation
import pickle
filename = 'finalized_model_naive_bayes.sav'
pickle.dump(gnb2, open(filename, 'wb'))

#just for testing the model object file
loaded_model = pickle.load(open(filename, 'rb'))
loaded_model.score(X_test,y_test)
