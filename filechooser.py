# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 01:30:31 2018

@author: jaide
"""
import tkinter as tk
import pickle
from tkinter.filedialog import askopenfilename
import sqlite3 
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import datetime
from flask import Flask,request,render_template,send_file,make_response,Response
from io import StringIO
app = Flask(__name__)

@app.route('/Test', methods=['GET', 'POST'])
def function():
    if request.method == 'POST':
        abspath = os.path.abspath(__file__)
        dname = os.path.dirname(abspath)
        print(dname)
        os.chdir(dname)


        root = tk.Tk()
    
        root.update()
        

        name = askopenfilename(title='Select data file',filetypes=[("CSV Files","*.csv")])
        root.withdraw()
        print(name)



    #filename = 'finalized_model.sav'
        loaded_model = pickle.load(open('finalized_model_clustering.sav', 'rb'))
        loaded_model1 = pickle.load(open('rf.sav', 'rb'))

#y_test=pd.read_csv('C:\\Users\\jaide\\pred.csv',header=None)
#X_test=pd.read_csv('C:\\Users\\jaide\\new_file.csv')
#X_test=X_test.drop('Unnamed: 0',axis=1)

#print(loaded_model.score(X_test,y_test))
        X_test=pd.read_csv(name)
        predictions=pd.DataFrame(columns=['Name','Predicted_position'])
        predictions['Name']=X_test.player_name
        predictions['Current_rating']=X_test.overall_rating
        cols=['gk_diving', 'finishing','standing_tackle','sliding_tackle','interceptions','marking','volleys','weight','strength','jumping']
        predictions['Predicted_position']=loaded_model.predict(X_test.filter(items=cols))
        predictions['Predicted_potential']=loaded_model1.predict(X_test.iloc[:,3:44])
        #pos_dict = { 3: 'Forward',  2: 'Midfielder',1: 'Defender', 0: 'Goalkeeper'}
        pos_dict = { 3: 'Midfielder',  2: 'Goalkeeper',1: 'Defender', 0: 'Forward',4: 'Midfielder'}
        predictions.loc[:, 'Predicted_position'] = predictions.loc[:, 'Predicted_position'].map(pos_dict)
        
        pot_dict = { 3: 'Reserve (70-80)',  2: 'Professional (80-90)',1: 'World Class (90+)' ,4: 'Amateur (60-70)',5: 'Novice (50-60)'}
        predictions.loc[:, 'Predicted_potential'] = predictions.loc[:, 'Predicted_potential'].map(pot_dict)
        X_test['Predicted_position']=predictions['Predicted_position']
        print(predictions)

        jsonfiles = json.loads(predictions.to_json(orient='records'))
        
        
        return render_template('index.html', ctrsuccess=jsonfiles) 
        
        #JSONP_data
        #return Response(predictions.to_csv())

    return '''<form method="POST">
                <style>
                    .wrapper {
                            max-width: 500px;
                            height: 100%;
                            margin: auto;
                            vertical-align: middle;
                            top: 50%;
                            padding: 10px;
                            }
                    .button {
                            position: absolute;
                            top: 50%;
                            left: 50%;
                            border-radius: 8px;
                            padding: 10px;
                            }
                </style>
                <head>
                    <title>Player Analytics</title>
                </head>
                <header>
                    <h1 align="center">Player Analytics</h1>
                    <h2 align="center">Select a CSV file containing Player Attributes for Analysis</h2>
                </header>    
                <body>
                <div class="wrapper">
                   <button class="button" color="blue">Select Input Data File</button>
                </div> 
                </body>
              </form>'''

if __name__ == "__main__":
    app.run()
#<input type="submit"  value="Select data file"><br>

