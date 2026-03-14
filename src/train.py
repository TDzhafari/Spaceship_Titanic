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


#######################################################################################################
#                                           Train Model
#######################################################################################################

def train_lr(X_train, y_train, preprocessor):
    lr_model = Pipeline([
    ("preprocessing", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
    ])
    lr_model.fit(X_train, y_train)
    return lr_model


def train_dt(X_train, y_train, preprocessor):
    dt = DecisionTreeClassifier(
    criterion = 'gini',
    min_samples_leaf=10,
    min_samples_split=2,
    max_depth=7,
    random_state=42
    )

    dt_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", dt)
    ])

    dt_model.fit(X_train, y_train)
    return dt_model

def train_rf(X_train, y_train, preprocessor):
    rf = RandomForestClassifier(
    n_estimators=300,
    min_samples_split=2,
    min_samples_leaf=1,
    max_depth=12,
    random_state=42,
    max_features='sqrt',
    n_jobs=-1
    )

    rf_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", rf)
    ])

    rf_model.fit(X_train, y_train)
    return rf_model

def train_xg(X_train, y_train, preprocessor):
    xgb = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
    )

    xgb_model = Pipeline([
        ("preprocessing", preprocessor),
        ("classifier", xgb)
    ])

    xgb_model.fit(X_train, y_train)
    return xgb_model

#######################################################################################################
#                                              Validate
#######################################################################################################

def validate(X_test, y_test, model):
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
#                                           Run Prediction
#######################################################################################################

if __name__ == '__main__':
    print('____________Start training___________')
    preprocessor = create_preprocessor()
    train_df, validate_df = fetch_data()
    validate_df = wrangle_data(validate_df)
    train_df = wrangle_data(train_df)
    X_train, X_test, y_train, y_test = split_data(train_df, True)

    print('__________Logistic Regression________')
    lr = train_lr(X_train, y_train, preprocessor)
    validate(X_test, y_test, lr)
    print('____________Decision Trees___________')
    dt = train_dt(X_train, y_train, preprocessor)
    validate(X_test, y_test, dt)
    print('____________Random Forest___________')
    rf = train_rf(X_train, y_train, preprocessor)
    validate(X_test, y_test, rf)
    print('______________XGBoost_______________')
    xg = train_xg(X_train, y_train, preprocessor)
    validate(X_test, y_test, xg)

    print('__________Training completed_________')