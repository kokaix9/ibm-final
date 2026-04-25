import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("model.pkl", "rb"))

st.title("Customer Churn Prediction")

st.write("Enter customer details:")


feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")

if st.button("Predict"):
    input_data = np.array([[feature1, feature2]])
    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("Customer will churn")
    else:
        st.success("Customer will not churn")
