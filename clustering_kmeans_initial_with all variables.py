# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 10:53:13 2018

@author: pragy
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np

train = pd.read_csv('F:/Player-Analytics/training set_new1.csv')
test = pd.read_csv('F:/Player-Analytics/test set_new1.csv')


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


from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=4, random_state=0)
kmeans.fit(train_new)
test['kmeans_2vars'] = kmeans.predict(test_new)

x=test.kmeans_2vars[test.position=='gk']
x.value_counts() # 2 for 1

y=test.kmeans_2vars[test.position=='def']
y.value_counts() # 1 for 2

z=test.kmeans_2vars[test.position=='mid']
z.value_counts()# 0 for 3

a=test.kmeans_2vars[test.position=='for']
a.value_counts() # 3 for 4

test['predicted5'] =np.where((test['kmeans_2vars']==2), 10,test['kmeans_2vars'])
test['predicted5'] =np.where((test['kmeans_2vars']==1), 20,test['predicted5'])
test['predicted5'] =np.where((test['kmeans_2vars']==0), 30,test['predicted5']) 
test['predicted5'] =np.where((test['kmeans_2vars']==3), 40,test['predicted5'])

test['predicted5'] =np.where((test['predicted5']==10), 1,test['predicted5'])
test['predicted5'] =np.where((test['predicted5']==20), 2,test['predicted5'])
test['predicted5'] =np.where((test['predicted5']==30), 3,test['predicted5'])
test['predicted5'] =np.where((test['predicted5']==40), 4,test['predicted5'])

#External Validation
test['position2'] =np.where((test['position']=='gk'), 1,test['position'])
test['position2'] =np.where((test['position2']=='def'), 2,test['position2'])
test['position2'] =np.where((test['position2']=='mid'), 3,test['position2'])
test['position2'] =np.where((test['position2']=='for'), 4,test['position2'])

test['position2']=test['position2'].astype('category')
test['predicted5']=test['predicted5'].astype('category')

from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
print(accuracy_score(test.predicted5,test.position2))#25.75
print(precision_score(test.predicted5,test.position2,average=None)) #[ 0.28873239  0.26153985  0.24806789  0.25657164]
print(recall_score(test.predicted5,test.position2,average=None)) #[ 0.09226961  0.32958419  0.37382964  0.23662645]
print(f1_score(test.predicted5,test.position2,average=None)) #[ 0.13984821  0.29164572  0.29823283  0.24619575]






