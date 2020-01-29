# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 17:12:26 2020

@author: hp
"""

import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('simple_linear_regression.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('test_home.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = request.form.to_dict()
    num = [int(int_features.get('x'))]
    #int_features = {1:2}
    #return int_features
    
    
    final_features = [np.array(num)]
    
    
    #return render_template('test.html', prediction_text=final_features)
        
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('front.html', prediction_text='your GAD score is $ {}'.format(output))


@app.route('/predict_api',methods=['POST'])
def predict_api():
    '''
    For direct API calls trought request
    '''
    data = request.get_json(force=True)
    prediction = model.predict([np.array(list(data.values()))])

    output = prediction[0]
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)