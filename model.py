import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score,
    roc_curve, confusion_matrix
)
from xgboost import XGBClassifier
from utils import load_data, preprocess_data, get_features_and_target

def train_model():
    # Load & preprocess
    df = load_data('data/loan_data.csv')
    df = preprocess_data(df)
    X, y = get_features_and_target(df)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric='logloss',
        random_state=42
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    print("=== Classification Report ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC Score: {roc_auc_score(y_test, y_prob):.4f}")

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/credit_model.pkl')
    joblib.dump(list(X.columns), 'models/feature_columns.pkl')
    print("✅ Model saved to models/credit_model.pkl")

    return model, X_test, y_test, y_prob, list(X.columns)

if __name__ == "__main__":
    train_model()