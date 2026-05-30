import streamlit as st
import joblib

# Load trained model
model = joblib.load("models/trained_model.pkl")

st.title("Predictive Analytics Dashboard")

value = st.number_input("Enter a value", min_value=0.0)

if st.button("Predict"):
    prediction = model.predict([[value]])
    st.success(f"Prediction: {prediction[0]}")