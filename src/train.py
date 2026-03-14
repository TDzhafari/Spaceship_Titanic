import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
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


#######################################################################################################
#                   Train Model
#######################################################################################################

def train_lr():
    pass

def train_dtree():
    pass

def train_rf():
    pass

def train_xgb():
    pass

#######################################################################################################
#                   Validate
#######################################################################################################

#######################################################################################################
#                   Run Prediction
#######################################################################################################

if __name__ == '__main__':
    print('_____Start training_____')
    preprocessor = create_preprocessor()
    train_df, validate_df = fetch_data()
    validate_df = wrangle_data(validate_df)
    train_df = wrangle_data(train_df)
    print(train_df.head(10).to_string())
    #X_train, X_test, y_train, y_test = split_data(train_df, True)
    print('_____Training Completed_____')