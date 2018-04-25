# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 13:46:31 2018

@author: pragy
"""

import matplotlib.pyplot as plt
import pandas as pd
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




#visualisation for predcited label
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

