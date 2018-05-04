# -*- coding: utf-8 -*-
"""
Created on Tue Apr 22 22:16:20 2018

@author: pragy
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import pandas as pd
import numpy as np
from scipy import linalg
import matplotlib as mpl
from sklearn import mixture
import seaborn as sns; sns.set()  # for plot styling

#Data retrieval
train = pd.read_csv('F:/Player-Analytics/training set_new1.csv')
train.position.value_counts()
test = pd.read_csv('F:/Player-Analytics/test set_new1.csv')
test.position.value_counts()
test.shape

#Data Manipulation for Unsupervised learning
train_new= train.drop('position',1)
train_new= train_new.drop('player_fifa_api_id',1)
train_new= train_new.drop('player_api_id',1)
train_new= train_new.drop('date',1)
train_new= train_new.drop('player_name',1)
train_new= train_new.drop('birthday',1)

#test_new= test.drop('position',1)
test_new= test.drop('player_fifa_api_id',1)
test_new= test_new.drop('player_api_id',1)
test_new= test_new.drop('date',1)
test_new= test_new.drop('player_name',1)
test_new= test_new.drop('birthday',1)

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

#preferred_foot
test_new = pd.DataFrame(test_new)
test_new.preferred_foot.value_counts()
test_new.preferred_foot = pd.factorize(test_new.preferred_foot)[0] #Right=1, left=0

#attacking_work_rate
attacking = {'medium': 0, 'high': 1, 'low': 2}
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='medium'), 0 ,test_new['attacking_work_rate'])
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='high'), 1 ,test_new['attacking_work_rate'])
test_new['attacking_work_rate'] =np.where((test_new['attacking_work_rate']=='low'), 2 ,test_new['attacking_work_rate'])

#defensive_work_rate

defensive = {'medium': 0, 'high': 2, 'low': 1}
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='medium'), 0 ,test_new['defensive_work_rate'])
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='high'), 1 ,test_new['defensive_work_rate'])
test_new['defensive_work_rate'] =np.where((test_new['defensive_work_rate']=='low'), 2 ,test_new['defensive_work_rate'])
# medium=0, high=2, low=1

#feature selection

train_new1 = train_new[['gk_diving', 'finishing','standing_tackle','sliding_tackle','interceptions','marking','volleys','weight','strength','jumping']].copy()
train_new1.shape
test_new1 = test_new[['gk_diving',  'finishing','standing_tackle','sliding_tackle','interceptions','marking','volleys','weight','strength','jumping']].copy()
test_new1.shape

corr = pd.DataFrame(test_new1.corr())
corr.to_csv('F:/Player-Analytics/correlation.csv')

#Modelling
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=5, random_state=0)
kmeans.fit(train_new1)
test['kmeans_2vars'] = kmeans.predict(test_new1)

x=test.kmeans_2vars[test.position=='gk']
x.value_counts() # 4 for 1

y=test.kmeans_2vars[test.position=='def']
y.value_counts() # 1 for 2

z=test.kmeans_2vars[test.position=='mid']
z.value_counts()# 2,3 for 3

a=test.kmeans_2vars[test.position=='for']
a.value_counts() # 0 for 4

#Assignment of labels to cluster based on domain knowledge
test['predicted5'] =np.where((test['kmeans_2vars']==4), 10,test['kmeans_2vars'])
test['predicted5'] =np.where((test['kmeans_2vars']==1), 20,test['predicted5'])
test['predicted5'] =np.where((test['kmeans_2vars']==2), 30,test['predicted5']) 
test['predicted5'] =np.where((test['kmeans_2vars']==3), 30,test['predicted5'])
test['predicted5'] =np.where((test['kmeans_2vars']==0), 40,test['predicted5'])

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
print(accuracy_score(test.predicted5,test.position2))#77.07
print(precision_score(test.predicted5,test.position2,average=None)) #[ 0.99798564  0.75363929  0.78810662  0.68571957]
print(recall_score(test.predicted5,test.position2,average=None)) #[ 0.9950227   0.84095337  0.66635787  0.81429198]
print(f1_score(test.predicted5,test.position2,average=None)) #[ 0.99650197  0.79490583  0.72213664  0.74449553]

test.to_csv('F:/Player-Analytics/test with predicted labels.csv')


#Adding overall visualisation code:

from sklearn.decomposition import PCA as sklearnPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets.samples_generator import make_blobs
from pandas.tools.plotting import parallel_coordinates

data = pd.read_csv('F:/Player-Analytics/test with predicted labels partial.csv')
list(data)
data.shape
data_new= data.drop('player_fifa_api_id',1)
data_new= data_new.drop('player_api_id',1)
data_new= data_new.drop('date',1)
data_new= data_new.drop('player_name',1)
data_new= data_new.drop('birthday',1)
data_new= data_new.drop('kmeans_2vars',1)
data_new= data_new.drop('position2',1)
data_new.describe()
list(data_new)

#Converting string categories to numerical
#preferred_foot
data_new = pd.DataFrame(data_new)
data_new.preferred_foot.value_counts()
data_new.preferred_foot = pd.factorize(data_new.preferred_foot)[0] #Right=1, left=0

#attacking_work_rate
data_new['attacking_work_rate'] =np.where((data_new['attacking_work_rate']=='medium'), 0 ,data_new['attacking_work_rate'])
data_new['attacking_work_rate'] =np.where((data_new['attacking_work_rate']=='high'), 1 ,data_new['attacking_work_rate'])
data_new['attacking_work_rate'] =np.where((data_new['attacking_work_rate']=='low'), 2 ,data_new['attacking_work_rate'])
data.attacking_work_rate.value_counts()
data_new.attacking_work_rate.value_counts()

#defensive_work_rate
data_new['defensive_work_rate'] =np.where((data_new['defensive_work_rate']=='medium'), 0 ,data_new['defensive_work_rate'])
data_new['defensive_work_rate'] =np.where((data_new['defensive_work_rate']=='high'), 1 ,data_new['defensive_work_rate'])
data_new['defensive_work_rate'] =np.where((data_new['defensive_work_rate']=='low'), 2 ,data_new['defensive_work_rate'])
# medium=0, high=2, low=1

#position
data_new['position'] =np.where((data_new['position']=='gk'), 0 ,data_new['position'])
data_new['position'] =np.where((data_new['position']=='def'), 1 ,data_new['position'])
data_new['position'] =np.where((data_new['position']=='mid'), 2 ,data_new['position'])
data_new['position'] =np.where((data_new['position']=='for'), 3 ,data_new['position'])
data.position.value_counts()

#visualisation for extracted label
list(data_new)
data_new= data_new.drop('predicted5',1)
y = data_new['position']          # Split off classifications
X = data_new.ix[:, 'defensive_work_rate':]

# Normalize the data attributes for the dataset.
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
pca = PCA(n_components=2)

# Fit and transform the data to the model
transformed = pd.DataFrame(pca.fit_transform(normalized_X))
transformed
fig = plt.figure()

plt.scatter(transformed[y==0][0], transformed[y==0][1], label='Goalkeepers', c='red')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Defenders', c='blue')
plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Midfielders', c='magenta')
plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Forward', c='yellow')
plt.legend()
plt.show()
fig.savefig('F:/Player-Analytics/test_test1.png')




#visualisation for predicted label
data_new= data_new.drop('position',1)
list(data_new)
y = data_new['predicted5']          # Split off classifications
X = data_new.loc[:, 'defensive_work_rate':]

# Normalize the data attributes for the dataset.
from sklearn import preprocessing
# normalize the data attributes
normalized_X = preprocessing.normalize(X)
pca = PCA(n_components=2)

# Fit and transform the data to the model
transformed = pd.DataFrame(pca.fit_transform(normalized_X))
transformed
fig = plt.figure()

plt.scatter(transformed[y==2][0], transformed[y==2][1], label='Defenders', c='blue')
plt.scatter(transformed[y==1][0], transformed[y==1][1], label='Goalkeepers', c='red')
plt.scatter(transformed[y==3][0], transformed[y==3][1], label='Midfielders', c='magenta')
plt.scatter(transformed[y==4][0], transformed[y==4][1], label='Forward', c='yellow')
plt.legend()
plt.show()
fig.savefig('F:/Player-Analytics/test_test2.png')



