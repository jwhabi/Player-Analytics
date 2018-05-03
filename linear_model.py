import numpy as np
import pandas as pd
import sqlite3 as sql
import os
import matplotlib.pyplot as plt
import datetime
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

project_path='C:\\vaibhav\\DPA'
os.chdir(project_path)

dropvars=['Unnamed: 0', 'player_fifa_api_id', 'player_api_id', 'date',
        'player_name', 'birthday','potential']

#Importing the train and test dataset which were generated from Data preprocessing step
train_base=pd.read_csv('training set.csv')
test_base=pd.read_csv('test set.csv')

from sklearn import linear_model as lm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error,classification_report,accuracy_score

#X = predictor variables of training set
X=train_base.drop(dropvars,axis=1)
#y = target variable of training set
y=pd.DataFrame(train_base['potential'])
#Xt = predictor variables of test set
Xt=test_base.drop(dropvars,axis=1)
#yt = target variable of test set
yt=pd.DataFrame(test_base['potential'])

#Normalizing(centering around 0 and scaling to unit variance)
#each columns in the predictor datasets
X_std=StandardScaler().fit_transform(X)
Xt_std=StandardScaler().fit_transform(Xt)

#Building, training, predicting on test set and evaluating the first linear model
lreg = lm.LinearRegression()
lreg.fit(X_std,y)
y_pred_lin=pd.DataFrame(lreg.predict(Xt_std))
print("Mean Squared Error: %.16f" % mean_squared_error(yt,y_pred_lin))   #MSE
print("R-square: %.16f" % r2_score(yt,y_pred_lin))             #R-square  
adj=((len(yt)*1.0)-1.0)*(1.0-r2_score(yt,y_pred_lin)) / ((len(yt)*1.0)-(len(lreg.coef_)*1.0)-1.0)
print("Adjusted R-square: %.16f" % adj) #Adjusted R-square


#Plotting the residuals for first 500 predictions and true values of test set 
#500 points are displayed as more than 10000 points on a graph would look like a 
#blob and not help us in gaining any understanding.
n=500
plt.scatter(y_pred_lin[:n], y_pred_lin[0][:n]-yt['potential'][:n], c='b', s=40,alpha=0.5)
plt.hlines(y=0,xmin=50,xmax=100)
plt.ylabel("Residuals")
plt.title("Residual plot for Linear regression model(test set)")

#Applying bucketing method
#This is being done to compare the evaluations of the linear model to RandomForest
def bucket(row):
    if row['potential']>=90:
        return 1
    elif (row['potential']>=80 and row['potential']<90):
        return 2
    elif row['potential']>=70 and row['potential']<80:
        return 3
    elif row['potential']>=60 and row['potential']<70:
        return 4
    return 5
def bucket_pred(row):
    if row[0]>=90:
        return 1
    elif (row[0]>=80 and row[0]<90):
        return 2
    elif row[0]>=70 and row[0]<80:
        return 3
    elif row[0]>=60 and row[0]<70:
        return 4
    return 5

yt_bucket=pd.DataFrame(yt.apply(lambda row: bucket(row), axis=1))
y_pred_bucket=pd.DataFrame(y_pred_lin.apply(lambda row: bucket_pred(row), axis=1))
print("Accuracy: %.16f" % accuracy_score(yt_bucket,y_pred_bucket))
print(classification_report(yt_bucket,y_pred_bucket))

###############################################################################
#Variable selection using Ridge
#Setting the range of alpha values(penalty)
alpha_ridge = [1e-5,1e-4,1e-3,1e-2,1e-1, 1, 5, 10]

#Building, training and checking the alpha value of the Ridge model
rreg_cv=lm.RidgeCV(alphas=alpha_ridge,scoring='r2')
rreg=rreg_cv.fit(X_std,y)
rreg.alpha_

#Predicting and evaluating the ridge model for best alpha value
y_pred_ridge=rreg.predict(Xt_std)
print("MSE: %.16f" % mean_squared_error(yt,y_pred_ridge))
print("R2: %.16f" % r2_score(yt,y_pred_ridge))

#Further investigation of Ridge model performance with alpha values
#Checking the Adjusted Rsquared value as well
#1e-5 is the best alpha value
for a in alpha_ridge:
    ridge_mod=lm.Ridge(alpha=a,normalize=True)
    ridge_mod.fit(X,y)
    pred=ridge_mod.predict(Xt)
    print("Alpha Value: %.5f" % a)
    print("MSE: %.16f" % mean_squared_error(yt,pred))
    print("R2: %.16f" % r2_score(yt,pred))
    print("Adj-R2: %.16f" % (len(yt)-1)*(1-r2_score(yt,pred)) / (len(yt)-len(ridge_mod.coef_)-1))
    print(ridge_mod.coef_.ravel()[:5])
    print('\n')

#Creating a table which could store feature names and their coefficients
coeff=pd.DataFrame(X.columns)
coeff['coefficients']=pd.Series(rreg.coef_.ravel())
sorted_vars=coeff.sort_values('coefficients',ascending=0)

#Plotting a bar graph to see different coefficient values
#Visually helps selecting variables
predictors = X.columns
coef = pd.Series(rreg.coef_.ravel(),predictors).sort_values()
coef.plot(kind='bar', title='Model Coefficients')
###############################################################################
#Linear model based on new variables

names=X.columns
lnr=rreg

#Evaluating the resulting linear model by using different sets of variables
#The threshold is set to programattically select variables above the threshold 
#value
for threshold in [0.1,0.2,0.3,0.4,0.5]:
    lst=list()
    for i in range(lnr.coef_.size):
        lst.append((names[i],round(lnr.coef_.item(i),3)))
    df=pd.DataFrame(lst)
    df.sort_values(1)
    #Selection of new varibles above the threshold value
    use_vars=np.where(abs(df[1])>=threshold)
    new_vars=df.iloc[use_vars][0]
    if threshold==0.4:#Choosing the threshold for the final set of variables to be used in linear model
        final_vars=new_vars
    
    #Normalizing the predictors for new set of variables
    new_X_std=StandardScaler().fit_transform(X[new_vars])
    new_Xt_std=StandardScaler().fit_transform(Xt[new_vars])
    
    #Building a linear model and checking the metrics
    new_lin=lm.LinearRegression()
    new_lin.fit(new_X_std,y)
    new_pred_lin=pd.DataFrame(new_lin.predict(new_Xt_std))
    print("Threshold: %.2f" % threshold)
    print("MSE: %.16f" % mean_squared_error(yt,new_pred_lin))
    print("R2: %.16f" % r2_score(yt,new_pred_lin))
    adj=(len(yt)-1)*(1-r2_score(yt,new_pred_lin)) / (len(yt)-len(new_lin.coef_)-1)
    print("Adj-R2: %.16f" % adj)
    print('\n')
    
    #Bucketing the predicted and true potential values to compare with Random forest model
    yt_bucket=pd.DataFrame(yt.apply(lambda row: bucket(row), axis=1))
    y_pred_bucket=pd.DataFrame(new_pred_lin.apply(lambda row: bucket_pred(row), axis=1))
    print("Accuracy: %.16f" % accuracy_score(yt_bucket,y_pred_bucket))
    print(classification_report(yt_bucket,y_pred_bucket))
    
###############################################################################
#Final set of variables with ridge model coefficients>=0.4
#We arrive at a set of 13 variables
#Same steps will be repeated as done to build the first linear model, albeit with 
#reduced number of variables
print("List of final variables")
print(final_vars)
fX_std=StandardScaler().fit_transform(X[final_vars])
fXt_std=StandardScaler().fit_transform(Xt[final_vars])
fmodel=lm.LinearRegression()
fmodel.fit(fX_std,y)
prediction=pd.DataFrame(fmodel.predict(fXt_std))
print("Mean Squared Error: %.16f" % mean_squared_error(yt,prediction))
print("R-square: %.16f" % r2_score(yt,prediction))
fadj=(len(yt)-1)*(1-r2_score(yt,prediction)) / (len(yt)-len(fmodel.coef_)-1)
print("Adjusted R-square: %.16f" % fadj)

#Plotting the residuals for first 500 predictions and true values of test set 
n=500
plt.scatter(prediction[:n], prediction[0][:n]-yt['potential'][:n], c='b', s=40,alpha=0.5)
plt.hlines(y=0,xmin=50,xmax=100)
plt.ylabel("Residuals")
plt.title("Residual plot for final Linear regression model(test set)")

#Applying bucketing method
yt_bucket=pd.DataFrame(yt.apply(lambda row: bucket(row), axis=1))
prediction_bucket=pd.DataFrame(prediction.apply(lambda row: bucket_pred(row), axis=1))
print("Accuracy: %.16f" % accuracy_score(yt_bucket,prediction_bucket))
print(classification_report(yt_bucket,prediction_bucket))
