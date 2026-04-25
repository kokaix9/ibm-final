import streamlit as st
import joblib
import pandas as pd

# Load model + features
model = joblib.load("model.pkl")
features = joblib.load("features.pkl")

st.title("Customer Churn Prediction")

st.write("Enter details:")

input_data = []

for feature in features:
    val = st.number_input(f"{feature}", value=0.0)
    input_data.append(val)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data], columns=features)
    prediction = model.predict(input_df)

    if prediction[0] == 1:
        st.error("Customer will churn ❌")
    else:
        st.success("Customer will NOT churn ✅")
