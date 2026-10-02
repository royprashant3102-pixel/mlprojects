from flask import Flask,request,render_template
import pandas as pd
import numpy as np


from sklearn.preprocessing import StandardScaler

application=Flak(__name__)
app=application

#route for home page

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods=['POST', 'GET'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html'):
    else:
        passs
        