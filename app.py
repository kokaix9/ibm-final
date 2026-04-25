import streamlit as st
import pandas as pd
import joblib

# =========================
# LOAD FILES
# =========================
model = joblib.load("model.pkl")
encoders = joblib.load("encoders.pkl")
features = joblib.load("features.pkl")

st.set_page_config(page_title="Churn Predictor", layout="centered")

st.title("✈️ Customer Churn Prediction")
st.write("Fill in the details below:")

# =========================
# INPUTS (MATCH YOUR DATASET)
# =========================

age = st.number_input("Age", 0, 100, 25)

frequent_flyer = st.selectbox(
    "Frequent Flyer",
    encoders['FrequentFlyer'].classes_
)

annual_income = st.selectbox(
    "Annual Income Class",
    encoders['AnnualIncomeClass'].classes_
)

services_opted = st.number_input("Services Opted", 0, 10, 1)

account_synced = st.selectbox(
    "Account Synced To Social Media",
    encoders['AccountSyncedToSocialMedia'].classes_
)

booked_hotel = st.selectbox(
    "Booked Hotel Or Not",
    encoders['BookedHotelOrNot'].classes_
)

# =========================
# ENCODE INPUT
# =========================

input_dict = {
    "Age": age,
    "FrequentFlyer": encoders['FrequentFlyer'].transform([frequent_flyer])[0],
    "AnnualIncomeClass": encoders['AnnualIncomeClass'].transform([annual_income])[0],
    "ServicesOpted": services_opted,
    "AccountSyncedToSocialMedia": encoders['AccountSyncedToSocialMedia'].transform([account_synced])[0],
    "BookedHotelOrNot": encoders['BookedHotelOrNot'].transform([booked_hotel])[0]
}

input_df = pd.DataFrame([input_dict])

# Ensure correct column order
input_df = input_df[features]

# =========================
# PREDICTION
# =========================

if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer will churn (Probability: {probability:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Probability: {probability:.2f})")
