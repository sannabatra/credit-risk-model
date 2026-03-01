import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def load_data(filepath):
    df = pd.read_csv(filepath)
    return df

def preprocess_data(df):
    df = df.copy()

    # Drop rows with too many nulls
    df.dropna(thresh=len(df.columns) - 2, inplace=True)

    # Fill missing numerical values with median
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col].fillna(df[col].median(), inplace=True)

    # Fill missing categorical values with mode
    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0], inplace=True)

    # Encode categorical columns
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])

    return df

def get_features_and_target(df, target_col='loan_status'):
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y