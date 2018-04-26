# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 13:44:35 2018

@author: pragy
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
import matplotlib as mpl
from sklearn import mixture


train = pd.read_csv('F:/Player-Analytics/training set_new1.csv')
train.position.value_counts()
test = pd.read_csv('F:/Player-Analytics/test set_new1.csv')
test.position.value_counts()
test.shape

#Data Manipulation
train_new= train.drop('position',1)
train_new= train_new.drop('player_fifa_api_id',1)
train_new= train_new.drop('player_api_id',1)
train_new= train_new.drop('date',1)
train_new= train_new.drop('player_name',1)
train_new= train_new.drop('birthday',1)

train_new.describe()

test_new= test.drop('position',1)
test_new= test_new.drop('player_fifa_api_id',1)
test_new= test_new.drop('player_api_id',1)
test_new= test_new.drop('date',1)
test_new= test_new.drop('player_name',1)
test_new= test_new.drop('birthday',1)

test_new.head(1)
test_new.describe()


#Converting string categories to numerical
#preferred_foot
train_new = pd.DataFrame(train_new)
train.preferred_foot.value_counts()
train_new.preferred_foot = pd.factorize(train_new.preferred_foot)[0] #Right=1, left=0

#attacking_work_rate
attacking = {'medium': 0, 'high': 1, 'low': 2}
train_new['attacking_work_rate'] =np.where((train_new['attacking_work_rate']=='medium'), 0 ,train_new['attacking_work_rate'])
train_new['attacking_work_rate'] =np.where((train_new['attacking_work_rate']=='high'), 1 ,train_new['attacking_work_rate'])
train_new['attacking_work_rate'] =np.where((train_new['attacking_work_rate']=='low'), 2 ,train_new['attacking_work_rate'])
train.attacking_work_rate.value_counts()
train_new.attacking_work_rate.value_counts()

#defensive_work_rate

defensive = {'medium': 0, 'high': 2, 'low': 1}
train_new['defensive_work_rate'] =np.where((train_new['defensive_work_rate']=='medium'), 0 ,train_new['defensive_work_rate'])
train_new['defensive_work_rate'] =np.where((train_new['defensive_work_rate']=='high'), 1 ,train_new['defensive_work_rate'])
train_new['defensive_work_rate'] =np.where((train_new['defensive_work_rate']=='low'), 2 ,train_new['defensive_work_rate'])
# medium=0, high=2, low=1
test_new.defensive_work_rate.value_counts()
test.defensive_work_rate.value_counts()


#preferred_foot
test_new = pd.DataFrame(test_new)
test_new.preferred_foot.value_counts()
test_new.preferred_foot = pd.factorize(test_new.preferred_foot)[0] #Right=1, left=0

#attacking_work_rate
attacking = {'medium': 0, 'high': 1, 'low': 2}
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='medium'), 0 ,test_new['attacking_work_rate'])
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='high'), 1 ,test_new['attacking_work_rate'])
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='low'), 2 ,test_new['attacking_work_rate'])
test.attacking_work_rate.value_counts()
test_new.attacking_work_rate.value_counts()

#defensive_work_rate

defensive = {'medium': 0, 'high': 2, 'low': 1}
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='medium'), 0 ,test_new['defensive_work_rate'])
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='high'), 1 ,test_new['defensive_work_rate'])
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='low'), 2 ,test_new['defensive_work_rate'])
# medium=0, high=2, low=1
test_new.defensive_work_rate.value_counts()
test.defensive_work_rate.value_counts()
train_new.shape
test_new.shape


'''for elem in train_new['defensive_work_rate'].unique():    
    train_new[str(elem)] = train_new['defensive_work_rate'] == elem

list(train_new)

for elem in test_new['defensive_work_rate'].unique():    
    test_new[str(elem)] = test_new['defensive_work_rate'] == elem'''

train_new1 = train_new[['gk_diving', 'standing_tackle', 'finishing','interceptions','marking','sliding_tackle','volleys','weight','strength','jumping']].copy()
train_new1.shape

test_new1 = test_new[['gk_diving', 'standing_tackle', 'finishing','interceptions','marking','sliding_tackle','volleys','weight','strength','jumping']].copy()
test_new1.shape


from sklearn.decomposition import RandomizedPCA,PCA

# Create a regular PCA model 
pca = PCA(n_components=5)
# Fit and transform the data to the model
train_new1 = pca.fit_transform(train_new1)
reduced_data_pca.shape
print(reduced_data_pca)
type(reduced_data_pca)
pca = PCA(n_components=5)
test_new1 = pca.fit_transform(test_new1)
test_new1.shape 

#Modelling
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(train_new1)
train['kmeans_2vars'] = kmeans.predict(train_new1)

train.position.value_counts()
train.kmeans_2vars.value_counts()

#Label Assignment
x=train.kmeans_2vars[train.position=='gk']
x.value_counts() # 4 for 1

y=train.kmeans_2vars[train.position=='def']
y.value_counts() # 0 for 2

z=train.kmeans_2vars[train.position=='mid']
z.value_counts()# 3,2 for 3

a=train.kmeans_2vars[train.position=='for']
a.value_counts() # 1 for 4

train.kmeans_2vars.value_counts()

train['predicted5'] =np.where((train['kmeans_2vars']==4), 10,train['kmeans_2vars'])
train['predicted5'] =np.where((train['kmeans_2vars']==0), 20,train['predicted5'])#changed
train['predicted5'] =np.where((train['kmeans_2vars']==3), 30,train['predicted5']) #changed
train['predicted5'] =np.where((train['kmeans_2vars']==2), 30,train['predicted5'])#changed
train['predicted5'] =np.where((train['kmeans_2vars']==1), 40,train['predicted5'])


train['predicted5'] =np.where((train['predicted5']==10), 1,train['predicted5'])
train['predicted5'] =np.where((train['predicted5']==20), 2,train['predicted5'])
train['predicted5'] =np.where((train['predicted5']==30), 3,train['predicted5'])
train['predicted5'] =np.where((train['predicted5']==40), 4,train['predicted5'])


#External Validation
train['position2'] =np.where((train['position']=='gk'), 1,train['position'])
train['position2'] =np.where((train['position2']=='def'), 2,train['position2'])
train['position2'] =np.where((train['position2']=='mid'), 3,train['position2'])
train['position2'] =np.where((train['position2']=='for'), 4,train['position2'])

train['position2']=train['position2'].astype('category')
train['predicted5']=train['predicted5'].astype('category')

from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
print(accuracy_score(train.predicted5,train.position2))#77.44
print(precision_score(train.predicted5,train.position2,average=None)) #[ 0.99798564  0.76874065  0.78827663  0.68097687]
print(recall_score(train.predicted5,train.position2,average=None)) #[ 0.9950227   0.84235242  0.67194293  0.81515889]
print(f1_score(train.predicted5,train.position2,average=None)) #[ 0.99650197  0.80386485  0.72547571  0.74205077]



