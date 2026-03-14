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

def train_lr(X_train, y_train, preprocessor):
    lr = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
    ])
    lr.fit(X_train, y_train)
    return lr


def train_dtree(X_train, y_train):
    pass

def train_rf(X_train, y_train):
    pass

def train_xgb(X_train, y_train):
    pass

#######################################################################################################
#                   Validate
#######################################################################################################

def validate(X_test, model):
    """
    Gets all relevant validation metrics and prints them
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)
    sensitivity = tp / (tp + fn)

    print("Logistic Regression Performance")
    print("-----------------")
    print(f"Accuracy     : {accuracy:.3f}")
    print(f"Precision    : {precision:.3f}")
    print(f"Recall       : {recall:.3f}")
    print(f"F1 Score     : {f1:.3f}")
    print(f"ROC AUC      : {roc_auc:.3f}")
    print(f"Sensitivity  : {sensitivity:.3f}")

    print("\nConfusion Matrix")
    print(cm)

    print(tn, fp, fn, tp)


    print("\nClassification Report")
    print(classification_report(y_test, y_pred))


#######################################################################################################
#                   Run Prediction
#######################################################################################################

if __name__ == '__main__':
    print('_____Start training_____')
    preprocessor = create_preprocessor()
    train_df, validate_df = fetch_data()
    validate_df = wrangle_data(validate_df)
    train_df = wrangle_data(train_df)
    X_train, X_test, y_train, y_test = split_data(train_df, True)
    lr = train_lr(X_train, y_train, preprocessor)
    validate(X_test, lr)
    print('_____Training Completed_____')