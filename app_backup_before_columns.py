import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("cad_model.pkl")

# Page Title

st.markdown("""
<div style="
background: linear-gradient(90deg, #0f172a, #1e3a8a);
padding:20px;
border-radius:15px;
text-align:center;
margin-bottom:20px;
">
<h2 style="color:white;">
🫀 AI-Powered CAD Risk Assessment
</h2>
<p style="color:#cbd5e1;">
Clinical Decision Support System using Machine Learning
</p>
</div>
""", unsafe_allow_html=True)


# Sidebar
st.sidebar.title("Project Information")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Accuracy: 86.89%")
st.sidebar.write("ROC-AUC: 0.9354")
st.sidebar.write("Dataset: Z-Alizadeh Sani")

st.sidebar.markdown("---")
st.sidebar.write("Developer: Tejasvi B")
st.sidebar.write("B.Tech Biotechnology")
st.sidebar.write("BIT, Sathyamangalam")

# Description
st.markdown("""
This application predicts the risk of Coronary Artery Disease (CAD)
using a Random Forest Machine Learning model trained on the
Z-Alizadeh Sani dataset.
""")

# User Inputs
age = st.number_input("Age", 20, 100, 50)
weight = st.number_input("Weight", 30, 150, 70)
bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
bp = st.number_input("Blood Pressure", 80, 200, 120)
fbs = st.number_input("FBS", 50, 300, 100)
tg = st.number_input("Triglycerides", 50, 500, 150)
hdl = st.number_input("HDL", 10, 100, 40)

sex = st.selectbox("Sex", ["Male", "Female"])
dm = st.selectbox("Diabetes", [0, 1])
htn = st.selectbox("Hypertension", [0, 1])
smoker = st.selectbox("Current Smoker", [0, 1])
ldl = st.number_input("LDL", 50, 300, 130)
chest_pain = st.selectbox("Typical Chest Pain", [0, 1])

# Prediction Button
if st.button("Predict CAD Risk"):

    with st.spinner("Analyzing patient data..."):

        patient = pd.DataFrame([{
            'Age': age,
            'Weight': weight,
            'Length': 170,
            'BMI': bmi,
            'DM': dm,
            'HTN': htn,
            'Current Smoker': smoker,
            'EX-Smoker': 0,
            'FH': 0,
            'BP': bp,
            'PR': 74,
            'Edema': 0,
            'Typical Chest Pain': chest_pain,
            'Function Class': 0,
            'Q Wave': 0,
            'St Elevation': 0,
            'St Depression': 0,
            'Tinversion': 0,
            'FBS': fbs,
            'CR': 0.9,
            'TG': tg,
            'LDL': ldl,
            'HDL': hdl,
            'BUN': 14,
            'ESR': 20,
            'HB': 14.3,
            'K': 4.7,
            'Na': 136,
            'WBC': 11300,
            'Lymph': 22,
            'Neut': 68,
            'PLT': 172,
            'EF-TTE': 50,
            'Region RWMA': 2,
            'Sex_Male': (sex == "Male"),
            'Obesity_Y': True,
            'CRF_Y': False,
            'CVA_Y': False,
            'Airway disease_Y': False,
            'Thyroid Disease_Y': False,
            'CHF_Y': False,
            'DLP_Y': False,
            'Weak Peripheral Pulse_Y': False,
            'Lung rales_Y': False,
            'Systolic Murmur_Y': False,
            'Diastolic Murmur_Y': False,
            'Dyspnea_Y': False,
            'Atypical_Y': False,
            'Nonanginal_Y': False,
            'LowTH Ang_Y': False,
            'LVH_Y': False,
            'Poor R Progression_Y': False,
            'VHD_N': False,
            'VHD_Severe': False,
            'VHD_mild': True
        }])

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        if prediction == 1:
            st.error("🚨 HIGH CAD RISK DETECTED")
            st.write(f"Confidence Score: {probability*100:.2f}%")
            st.metric("Prediction Confidence", f"{probability*100:.2f}%")

            if probability < 0.70:
                st.warning("Risk Level: Moderate")
            else:
                st.warning("Risk Level: High")

        else:
            st.success("✅ LOW CAD RISK")
            st.write(f"Confidence Score: {(1-probability)*100:.2f}%")
            st.metric("Prediction Confidence", f"{(1-probability)*100:.2f}%")
            st.info("Risk Level: Low")

        # Patient Summary
        st.success("📋 Patient Summary")

        st.write(f"**Age:** {age}")
        st.write(f"**Sex:** {sex}")
        st.write(f"**Diabetes:** {'Yes' if dm == 1 else 'No'}")
        st.write(f"**Hypertension:** {'Yes' if htn == 1 else 'No'}")
        st.write(f"**Current Smoker:** {'Yes' if smoker == 1 else 'No'}")
        st.write(f"**LDL:** {ldl}")

        st.markdown("---")

# SHAP Explainability
st.subheader("Model Explainability")
st.image("shap_summary.png")

# Disclaimer
st.warning(
    "This tool is for educational and research purposes only and should not be used as a substitute for professional medical advice."
)