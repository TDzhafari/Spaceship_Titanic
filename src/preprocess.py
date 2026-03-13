import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

#######################################################################################################
#   Project: Spaceship Titatic
#   Author: Timur Dzhafari
#   Date: (TBD)
#
#   Description: 
#######################################################################################################




#######################################################################################################
#                   Exploratory Data Analysis
#######################################################################################################

def fetch_data():

    train_df = pd.read_csv(r'C:\Users\gagar\OneDrive\Documents\GitHub\Spaceship_Titanic\data\test.csv')
    test_df = pd.read_csv(r'C:\Users\gagar\OneDrive\Documents\GitHub\Spaceship_Titanic\data\test.csv')

    return train_df, test_df

def run_eda(train_df):

    # statistical overview of the dataset
    train_df.describe()

    # check missing values
    train_df.isnull().sum()

#######################################################################################################
#                   Data Prep
#######################################################################################################

def wrangle_data(train_df):
    train_df['VIP'] = train_df['VIP'].astype('str')
    train_df['VIP'] = train_df[train_df['VIP'] == 'nan'] = 'Unknown'
    train_df['CryoSleep'] = train_df[train_df['CryoSleep'] == 'nan'] = 'Unknown'
    train_df['Destination'] = train_df['Destination'].astype('str')
    train_df['CryoSleep'] = train_df['CryoSleep'].astype('str')


    numerical_fields = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Age']
    categorical_fields = ['HomePlanet','CryoSleep','Destination', 'VIP','cabin_name_1', 'cabin_name_3']
    #string_fields = []

    numerical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")), #imputing nan values setting median for skewed datasets
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    string_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_fields),
        ("cat", categorical_pipeline, categorical_fields),
    #    ("str", string_pipeline, string_fields)
    ], remainder='passthrough')

    return preprocessor

#######################################################################################################
#                   Train Model
#######################################################################################################

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
    
    train_df, test_df = fetch_data()
    run_eda(train_df)
    wrangle_data(train_df)