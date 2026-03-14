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

#######################################################################################################
#   Project: Spaceship Titatic (Preprocessing)
#   Author: Timur Dzhafari
#   Date: (TBD)
#
#   Description: 
#######################################################################################################




#######################################################################################################
#                   Exploratory Data Analysis
#######################################################################################################

def fetch_data():

    train_df = pd.read_csv(r'C:\Users\gagar\OneDrive\Documents\GitHub\Spaceship_Titanic\data\train.csv')
    test_df = pd.read_csv(r'C:\Users\gagar\OneDrive\Documents\GitHub\Spaceship_Titanic\data\test.csv')

    return train_df, test_df

def run_eda(df):

    # statistical overview of the dataset
    print('____Statistical overview of the numerical values in the dataset:_____')
    print(df.describe())

    # check missing values
    print('____Checking for null values____')
    print(df.isnull().sum())

    print('____Reviewing column datatypes____')
    print(df.isnull().sum())

#######################################################################################################
#                   Data Prep
#######################################################################################################

def wrangle_data(df):
    """
    Wrangling the dataset, preparing it for modeling
    """
    # Replacing missing values with comparable structure cabin ids
    df.Cabin = df.Cabin.fillna('Unknown/Unknown/Unknown')
    # Feature eangineering cabin id into 3 usable fields
    df['cabin_name_1'] = df['Cabin'].apply(lambda x: x.split('/')[0])
    df['cabin_name_2'] = df['Cabin'].apply(lambda x: x.split('/')[1])
    df['cabin_name_3'] = df['Cabin'].apply(lambda x: x.split('/')[2])   
    # Fixing missing VIP values
    df['VIP'] = df['VIP'].astype('str')
    df['VIP'] = df[df['VIP'] == 'nan'] = 'Unknown'
    # Updating datatype for destination
    df['Destination'] = df['Destination'].astype('str')
    # Imputing cryosleep missing values
    df['CryoSleep'] = df[df['CryoSleep'] == 'nan'] = 'Unknown'
    df['CryoSleep'] = df['CryoSleep'].astype('str')

    return df

def create_preprocessor():
    
    numerical_fields = ['RoomService','FoodCourt','ShoppingMall','Spa','VRDeck','Age']
    categorical_fields = ['HomePlanet','CryoSleep','Destination', 'VIP','cabin_name_1', 'cabin_name_3']

    numerical_pipeline = Pipeline([

        ("imputer", SimpleImputer(strategy="median")), #imputing nan values setting median for skewed datasets
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(sparse_output=False, handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("num", numerical_pipeline, numerical_fields),
        ("cat", categorical_pipeline, categorical_fields),
    ], remainder='passthrough')

    return preprocessor

def split_data(df, do_split=True):
    """
    Not entirely necessary 
    """
    print(df.columns)
    x = df.drop(columns=["PassengerId", "Transported", "cabin_name_2",
                         "Cabin", "Name"])
    df["Transported"] = df["Transported"].astype('float')
    y = df["Transported"]
    if do_split:
        X_train, X_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
        )   

        return X_train, X_test, y_train, y_test
    return x, y

#######################################################################################################
#                   Run Prediction
#######################################################################################################

if __name__ == '__main__':
    
    train_df, test_df = fetch_data()
    run_eda(train_df)
    df = wrangle_data(train_df)
    print(df.head(10).to_string())

    print('__________________________Preprocessing complete________________________')