# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 21:17:24 2018

@author: abans
"""

import pandas as pd
import os
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree, metrics
import random
project_path='C:\\Users\\abans\\Documents\\DPA MATH 571\\Project\\Soccer Data'
os.chdir(project_path)
random.seed(50)
tree_train_raw=pd.read_csv('training set.csv')
tree_test_raw=pd.read_csv('test set.csv')
#Creating buckets for the target variable
def buckets (row):
    if row['potential'] >= 90 :
        return 1
    if row['potential'] >= 80 :
        return 2
    if row['potential'] >= 70 :
        return 3
    if row['potential'] >= 60 :
        return 4
    return 5
tree_train_raw.head()
tree_train_raw['potential_bucket']=tree_train_raw.apply(lambda row: buckets(row), axis=1)
tree_train_raw['potential_bucket']=tree_train_raw['potential_bucket'].astype('category')
tree_train_raw['position']=tree_train_raw['position'].astype('category')
tree_test_raw['potential_bucket']=tree_test_raw.apply(lambda row: buckets(row), axis=1)
tree_test_raw['potential_bucket']=tree_test_raw['potential_bucket'].astype('category')
tree_test_raw['position']=tree_test_raw['position'].astype('category')
tree_train_raw['potential_bucket'].dtypes
tree_train_raw.dtypes
type(tree_train_raw)
#Dropping irrelevant variables
dropvars=['Unnamed: 0',
 'player_fifa_api_id',
 'player_api_id',
 'date',
 'player_name',
 'birthday',
 'potential']
random.seed(50)
tree_train=tree_train_raw.drop(dropvars, axis=1)
tree_test=tree_test_raw.drop(dropvars, axis=1)

#creating dataframes for independent and dependent variables
predict_train=pd.DataFrame(tree_train_raw['potential_bucket'])
predict_test=pd.DataFrame(tree_test_raw['potential_bucket'])
tree_train=tree_train.drop('potential_bucket', axis=1)
tree_test=tree_test.drop('potential_bucket', axis=1)
accuracy = pd.DataFrame(columns=['Depth', 'Leaf', 'Accuracy'])
#Creating a loop to test different values for max leaf nodes and depth of the decision tree
for n in range(5,16):
   for k in (5, 10, 20,30):
       clf=tree.DecisionTreeClassifier(max_depth=n, max_leaf_nodes=k)
       clf=clf.fit(tree_train,predict_train)
       acu=metrics.accuracy_score(predict_test, clf.predict(tree_test))
       print (n, " ",k, " ",acu)
       accuracy=accuracy.append([{'Depth':n,'Leaf':k,'Accuracy':acu}])
#Creating a decision tree for max depth 5 and max leaf nodes 30
clf=tree.DecisionTreeClassifier(max_depth=5, max_leaf_nodes=30)
clf=clf.fit(tree_train,predict_train)
acu=metrics.accuracy_score(predict_test, clf.predict(tree_test))
feature_imp=pd.DataFrame(tree_train.columns,clf.feature_importances_)
feature_imp.sort_values
feature_imp.shape
type(clf.feature_importances_)

#Only important variables are Overall Rating,interceptions, gk_kicking, age
impvars=['overall_rating','interceptions', 'gk_kicking', 'age']
predict_train2=pd.DataFrame(tree_train_raw['potential_bucket'])
predict_test2=pd.DataFrame(tree_test_raw['potential_bucket'])
tree_train2=tree_train[impvars]
tree_test2=tree_test[impvars]
clf=tree.DecisionTreeClassifier(max_depth=3)   
clf=clf.fit(tree_train2,predict_train2)
acu=metrics.accuracy_score(predict_test, clf.predict(tree_test))
feature_imp=pd.DataFrame(tree_train2.columns,clf.feature_importances_)
feature_imp.sort_values
feature_imp.shape
type(clf.feature_importances_)
acu
#exporting the previously created decision tree using Graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,
                feature_names=tree_train.columns,
                class_names=["1", "2", "3", "4", "5"])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())

#Creating random forest classifier
from sklearn.ensemble import RandomForestClassifier
rforest1 = RandomForestClassifier(max_depth=20, random_state=0)
rforest1.fit(tree_train, predict_train)
acuforest=metrics.accuracy_score(predict_test, rforest1.predict(tree_test))
classforest=metrics.classification_report(predict_test, rforest1.predict(tree_test))
print(acuforest)
#exporting the decision tree as an object using pickle
import pickle
filename = 'finalized_model.sav'
pickle.dump(rforest1, open(filename, 'wb'))



























