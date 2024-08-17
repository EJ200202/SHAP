import streamlit as st
import shap
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from streamlit_shap import st_shap

# Title and Introduction
st.title("Customer Churn Prediction with SHAP")
st.write("""
This app allows you to explore how the RandomForestClassifier model predicts customer churn 
and understand which features contribute most to the model's decisions.
""")

# Load Dataset
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
if uploaded_file is not None:
    customer = pd.read_csv(uploaded_file)
else:
    # For the purpose of the example, load a default dataset if no file is uploaded
    customer = pd.read_csv("Customer Churn.csv")  # Replace with your default file path

# Preprocessing
X = customer.drop("Churn", axis=1)
y = customer.Churn
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

# Model Training
clf = RandomForestClassifier()
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# SHAP Explainer
explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test)

# Part 1: General SHAP Analysis
st.header("Part 1: General SHAP Analysis")

# Display Classification Report
st.subheader("Classification Report")
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(report)

# SHAP Summary Plot
st.subheader("SHAP Summary Plot")
fig, ax = plt.subplots()
shap.summary_plot(shap_values, X_test, show=False)
st.pyplot(fig)

# Part 2: Individual Input Prediction & Explanation
st.header("Part 2: Individual Input Prediction & Explanation")

# Dropdown to Select Data Point
st.subheader("Force Plot")
index = st.slider("Select data point index", 0, len(X_test) - 1, 0)
st_shap(shap.force_plot(explainer.expected_value[0], shap_values[0][index], X_test.iloc[index]), height=400, width=1000)

# SHAP Decision Plot
st.subheader("Decision Plot")
st_shap(shap.decision_plot(explainer.expected_value[0], shap_values[0], X_test.iloc[index:index+1]))

# Interactive Feature Adjustment
st.subheader("Adjust Feature Values and Predict")
input_data = {}
for feature in X.columns:
    input_data[feature] = st.slider(f"Adjust {feature}", float(X[feature].min()), float(X[feature].max()), float(X[feature].mean()))

# Convert input data to DataFrame
input_df = pd.DataFrame(input_data, index=[0])

# Predict on adjusted features
prediction = clf.predict(input_df)[0]
probability = clf.predict_proba(input_df)[0][1]  # Probability of churn

# Display Prediction and Probability
st.write(f"**Prediction:** {'Churn' if prediction == 1 else 'No Churn'}")
st.write(f"**Churn Probability:** {probability:.2f}")

# SHAP Explanation for the Adjusted Input
shap_values_input = explainer.shap_values(input_df)

# Display Force Plot for Adjusted Input
st.subheader("Force Plot for Adjusted Input")
st_shap(shap.force_plot(explainer.expected_value[0], shap_values_input[0], input_df), height=400, width=1000)

# Insights and Interpretation
st.header("Insights and Interpretation")
st.write("""
The SHAP analysis reveals that features such as [mention key features] have a significant impact on the churn prediction. 
We observed that [describe specific interactions or insights]. 
This helps in understanding the model's decision-making process and identifying areas where the model might be biased or limited.
""")

