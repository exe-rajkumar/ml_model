from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle

app = Flask(__name__)

# Load the trained model and TF-IDF vectorizer
with open('model.pkl', 'rb') as file:
    feature_extraction, model = pickle.load(file)

@app.route('/')
def home():
    return 'Home Page'

@app.route('/predict', methods=['POST'])
def predict():
    message = request.form['message']
    input_data_features = feature_extraction.transform([message])
    prediction = model.predict(input_data_features)
    print(prediction)
    if prediction[0] == 1:
        result = 1
    else:
        result = 0

    return result

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=8080)
