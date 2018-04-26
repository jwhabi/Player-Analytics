# -*- coding: utf-8 -*-
"""
Created on Wed Apr 11 13:26:49 2018

@author: abans
"""
from matplotlib import pyplot as plot
import seaborn as sns
import pandas as pd
import os
import numpy as np
project_path='C:\\Users\\abans\\Documents\\DPA MATH 571\\Project\\Soccer Data'
os.chdir(project_path)
player_data=pd.read_csv("player_data.csv")
player_data.head()
player_data.dtypes
player_data=player_data.drop(["Unnamed: 0", "player_fifa_api_id", "player_api_id"],axis=1)
player_data.columns
corr=player_data.corr()
corr.to_csv('correlation.csv')
print(corr)
fig1,ax1 = plot.subplots(nrows = 1,ncols = 1)
fig1.set_size_inches(w=30,h=24)
#sns.heatmap(corr,annot = True,ax = ax2, cmap="YlGnBu", cbar_kws={"orientation": "horizontal"})
mask = np.zeros_like(corr)
mask[np.triu_indices_from(mask)] = True
with sns.axes_style("white"): ax = sns.heatmap(corr.abs(), annot = True, mask=mask, square=True,cmap="YlGnBu")

#Creating Histograms
player_datahist=player_data.drop(["preferred_foot", "attacking_work_rate", "defensive_work_rate"],axis=1)
player_datahist.head()
pat = player_datahist.loc[:,player_datahist.columns.tolist()]
fighist, ax2 = plot.subplots(nrows=7,ncols=6)
fighist.set_size_inches(20,16)
for i,j in enumerate(player_datahist.select_dtypes(include = ['float64','int64']).columns.tolist()):
    sns.distplot(pat.loc[:,j],kde = False,hist = True, ax = ax2[int(i/6)][i%6])
fighist.tight_layout()

#Creating Boxplot
figbox, ax2 = plot.subplots(nrows=7,ncols=6)
figbox.set_size_inches(20,16)
for i,j in enumerate(player_datahist.select_dtypes(include = ['float64','int64']).columns.tolist()):
    sns.boxplot(pat.loc[:,j], ax = ax2[int(i/6)][i%6])
figbox.tight_layout()

