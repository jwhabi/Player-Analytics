import pandas as pd
import os
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn import tree, metrics

project_path='C:\\Users\\abans\\Documents\\DPA MATH 571\\Project\\Soccer Data'
os.chdir(project_path)

tree_train_raw=pd.read_csv('training set.csv')
tree_test_raw=pd.read_csv('test set.csv')

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
tree_test_raw['potential_bucket']=tree_test_raw.apply(lambda row: buckets(row), axis=1)
tree_test_raw['potential_bucket']=tree_test_raw['potential_bucket'].astype('category')

tree_train_raw['potential_bucket'].dtypes

dropvars=['Unnamed: 0',
 'player_fifa_api_id',
 'player_api_id',
 'date',
 'player_name',
 'birthday',
 'potential']


tree_train=tree_train_raw.drop(dropvars, axis=1)
#tree_train_base=tree_train_base.drop(tree_train_base.columns[1:39], axis=1)
tree_test=tree_test_raw.drop(dropvars, axis=1)
#tree_test_base=tree_test_base.drop(tree_test_base.columns[1:39], axis=1)

predict_train=pd.DataFrame(tree_train_raw['potential_bucket'])
predict_test=pd.DataFrame(tree_test_raw['potential_bucket'])
tree_train=tree_train.drop('potential_bucket', axis=1)#to drop the rating columns
tree_test=tree_test.drop('potential_bucket', axis=1)
accuracy = pd.DataFrame(columns=['Depth', 'Leaf', 'Accuracy'])
for n in range(5,16):
    for k in (5, 10, 20,30):
#        print("Preparing decision tree for n=",n)
        clf=tree.DecisionTreeClassifier(max_depth=n, max_leaf_nodes=k)
        clf=clf.fit(tree_train,predict_train)
#        print("Decision tree built. Calculating accuracy\n")
        acu=metrics.accuracy_score(predict_test, clf.predict(tree_test))
        print (n, " ",k, " ",acu)
        accuracy=accuracy.append([{'Depth':n,'Leaf':k,'Accuracy':acu}])


accuracy.sort_values(by=['Accuracy'], ascending = False)

#        accuracy.loc[n:1] = n
#        accuracy.loc[n:2] = k
#        accuracy.loc[n:1] = metrics.accuracy_score(predict_test, clf.predict(tree_test))
        
from sklearn.ensemble import RandomForestClassifier
rforest1 = RandomForestClassifier(max_depth=20, random_state=0)
rforest1.fit(tree_train, predict_train)
acuforest=metrics.accuracy_score(predict_test, rforest1.predict(tree_test))
print(acuforest)
