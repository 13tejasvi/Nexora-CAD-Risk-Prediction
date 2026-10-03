import streamlit as st
import joblib
import pandas as pd

model = joblib.load("cad_model.pkl")

st.title("🫀 CAD Risk Prediction System")

age = st.number_input("Age", 20, 100, 50)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
bp = st.number_input("Blood Pressure", 80, 200, 120)
fbs = st.number_input("FBS", 50, 300, 100)
tg = st.number_input("Triglycerides", 50, 500, 150)
hdl = st.number_input("HDL", 10, 100, 40)

st.write("Patient Data Entered Successfully")
if st.button("Predict CAD Risk"):
    st.success("Prediction button clicked!")