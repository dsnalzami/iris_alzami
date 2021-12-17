# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from flask import Flask, request, render_template,jsonify, url_for
import numpy as np
import joblib
import time
import os

app = Flask(__name__)
# Loading the model
model_file = open('finalized_model.sav', 'rb')
model = joblib.load(model_file)
labels = {1:"Iris-setosa", 2:"Iris-versicolor", 3:"Iris-virginica"}

# TODO: add versioning to url
@app.route('/', methods=['GET', 'POST'])
def predict():
    """ Main webpage with user input through form and prediction displayed

    :return: main webpage host, displays prediction if user submitted in text field
    """

    if request.method == 'POST':
        
        SepalLength = float(request.form['SepalLength'])
        SepalWidth  = float(request.form['SepalWidth'])
        PetalLength = float(request.form['PetalLength'])
        PetalWidth  = float(request.form['PetalWidth'])
        
        # Converting the inputs into a numpy array
        pred_args = np.array([SepalLength, SepalWidth, PetalLength, PetalWidth]).reshape(1, -1)
        
        # Predicting the Label
        model_prediction = model.predict(pred_args)[0]
        #prediction = labels[model_prediction]

        return render_template('index.html', text=model_prediction, SepalLength=SepalLength, SepalWidth=SepalWidth, PetalLength=PetalLength, PetalWidth=PetalWidth)

    if request.method == 'GET':
        return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)