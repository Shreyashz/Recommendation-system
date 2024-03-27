from flask import Flask, request, jsonify
import numpy as np
import requests
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import json
from .routes import Main
import joblib

# Load MinMaxScaler model
def create_app():
    app = Flask(__name__)
    scaler = joblib.load('scaler_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    tfidf_matrix = joblib.load('tfidf_matrix.pkl')
    df=pd.read_csv('Final_data.csv')
    app.register_blueprint(Main)
    return app