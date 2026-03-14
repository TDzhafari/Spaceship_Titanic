import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from preprocess import *

#######################################################################################################
#   Project: Spaceship Titatic (Train)
#   Author: Timur Dzhafari
#   Date: (TBD)
#
#   Description: 
#######################################################################################################

def load_model(model_path="models/spaceship_titanic_model.pkl"):
    """
    Load XGBoost 
    """
    model = joblib.load(model_path)
    return model

def predict(model, x):
    """
    Run inference using new data
    """
    pred = model.predict(x)
    proba = model.predict_proba(x)[:, 1]

    results = x.copy()
    results["prediction"] = pred
    results["probability"] = proba

    return results

if __name__ == "__main__":
    model = load_model()
    sample_data = pd.DataFrame({
        "HomePlanet": ["Earth"],
        "CryoSleep": [False],
        "Cabin": ["G/0/P"],
        "Destination": ["TRAPPIST-1e"],
        "Age": [19],
        "VIP": [False],
        "RoomService": [0],
        "FoodCourt": [0],
        "ShoppingMall": [0],
        "Spa": [0],
        "VRDeck": [0]
    })
    sample_data = wrangle_data(sample_data)
    predictions = predict(model, sample_data)
    print("\nPrediction results:")
    print(predictions)