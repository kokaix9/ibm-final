import streamlit as st
import pandas as pd
import pickle

# Load model and encoders
model = pickle.load(open("rf_model.pkl", "rb"))
encoders = pickle.load(open("label_encoders.pkl", "rb"))

st.title("✈️ Customer Churn Prediction")

st.write("Enter customer details:")

# INPUTS
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

# ENCODE INPUT
input_data = {
    "Age": age,
    "FrequentFlyer": encoders['FrequentFlyer'].transform([frequent_flyer])[0],
    "AnnualIncomeClass": encoders['AnnualIncomeClass'].transform([annual_income])[0],
    "ServicesOpted": services_opted,
    "AccountSyncedToSocialMedia": encoders['AccountSyncedToSocialMedia'].transform([account_synced])[0],
    "BookedHotelOrNot": encoders['BookedHotelOrNot'].transform([booked_hotel])[0]
}

input_df = pd.DataFrame([input_data])

# PREDICTION
if st.button("Predict"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"❌ Customer will churn (Prob: {prob:.2f})")
    else:
        st.success(f"✅ Customer will NOT churn (Prob: {prob:.2f})")
