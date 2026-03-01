import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from sklearn.metrics import roc_curve, roc_auc_score
from utils import load_data, preprocess_data, get_features_and_target
from sklearn.model_selection import train_test_split

# Auto-train model if not exists
import os
if not os.path.exists('models/credit_model.pkl'):
    from model import train_model
    train_model()


# --- Page Config ---
st.set_page_config(
    page_title="Credit Risk Analyzer",
    page_icon="💳",
    layout="wide"
)

# --- Load Model ---
@st.cache_resource
def load_model():
    model = joblib.load('models/credit_model.pkl')
    features = joblib.load('models/feature_columns.pkl')
    return model, features

@st.cache_data
def load_and_prepare_data():
    df = load_data('data/loan_data.csv')
    df = preprocess_data(df)
    X, y = get_features_and_target(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_test, y_test

# --- Header ---
st.title("💳 AI-Powered Credit Risk Assessment")
st.markdown("*Predicting loan default probability using Machine Learning*")
st.divider()

model, feature_cols = load_model()
X_test, y_test = load_and_prepare_data()

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🔍 Predict Default Risk", "📊 Model Performance", "📈 Data Insights"])

# TAB 1 - Prediction
# TAB 1 - Prediction
with tab1:
    st.subheader("Enter Borrower Details")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.slider("Age", 18, 75, 35)
        income = st.number_input("Annual Income ($)", 10000, 500000, 60000, step=1000)
        emp_length = st.slider("Employment Length (years)", 0, 30, 5)
        cred_hist_length = st.slider("Credit History Length (years)", 0, 30, 5)

    with col2:
        loan_amount = st.number_input("Loan Amount ($)", 1000, 100000, 15000, step=500)
        loan_int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
        loan_percent_income = round(loan_amount / income, 2)
        st.metric("Loan-to-Income Ratio", loan_percent_income)

    with col3:
        loan_grade = st.selectbox("Loan Grade", [0,1,2,3,4,5,6],
                                   format_func=lambda x: ['A','B','C','D','E','F','G'][x])
        loan_intent = st.selectbox("Loan Intent", [0,1,2,3,4,5],
                                    format_func=lambda x: ['Personal','Education','Medical','Venture','Home Improvement','Debt Consolidation'][x])
        cb_default = st.selectbox("Previous Default?", [0,1],
                                   format_func=lambda x: "No" if x == 0 else "Yes")
        home_ownership = st.selectbox("Home Ownership", [0,1,2,3],
                                       format_func=lambda x: ['Rent','Own','Mortgage','Other'][x])

    if st.button("🔮 Predict Default Risk", use_container_width=True):
        input_data = pd.DataFrame([[
            age, income, home_ownership, emp_length,
            loan_intent, loan_grade, loan_amount,
            loan_int_rate, loan_percent_income,
            cb_default, cred_hist_length
        ]], columns=feature_cols)

        prob = model.predict_proba(input_data)[0][1]
        risk_label = "🔴 High Risk" if prob > 0.5 else "🟢 Low Risk"

        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Default Probability", f"{prob*100:.1f}%")
        c2.metric("Risk Category", risk_label)
        c3.metric("Confidence", f"{max(prob, 1-prob)*100:.1f}%")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={'text': "Default Risk Score"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "darkred" if prob > 0.5 else "green"},
                'steps': [
                    {'range': [0, 40], 'color': "#d4edda"},
                    {'range': [40, 70], 'color': "#fff3cd"},
                    {'range': [70, 100], 'color': "#f8d7da"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
# TAB 2 - Model Performance
with tab2:
    st.subheader("Model Performance Metrics")

    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)

    col1, col2 = st.columns(2)

    with col1:
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name=f'AUC = {auc:.3f}', fill='tozeroy'))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name='Random', line=dict(dash='dash')))
        fig_roc.update_layout(title='ROC Curve', xaxis_title='False Positive Rate',
                               yaxis_title='True Positive Rate')
        st.plotly_chart(fig_roc, use_container_width=True)

    with col2:
        importance = model.feature_importances_
        fig_imp = px.bar(
            x=importance, y=feature_cols,
            orientation='h', title='Feature Importance',
            labels={'x': 'Importance', 'y': 'Feature'}
        )
        fig_imp.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    st.metric("AUC-ROC Score", f"{auc:.4f}")

# TAB 3 - Data Insights
with tab3:
    st.subheader("Dataset Overview")
    df_raw = load_data('data/loan_data.csv')
    st.write(f"Total Records: **{len(df_raw):,}**")
    st.dataframe(df_raw.head(20), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df_raw, x='loan_status', title='Default vs Non-Default Distribution',
                           color='loan_status')
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.histogram(df_raw, x='loan_int_rate', title='Interest Rate Distribution',
                           nbins=30, color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)