import streamlit as st
import joblib
import pandas as pd

# ------------------ CUSTOM STYLING ------------------

st.markdown("""
<style>

/* Main background */
.stApp {
    background-color: #0b1220;
}

/* Buttons */
.stButton > button {
    border-radius: 12px;
    height: 50px;
    font-weight: bold;
    width: 100%;
}

/* Metric cards */
[data-testid="stMetric"] {
    background-color: #111827;
    border: 1px solid #374151;
    padding: 15px;
    border-radius: 12px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #1b1f2f;
}

/* Alert boxes */
[data-testid="stAlert"] {
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------

model = joblib.load("cad_model.pkl")


# ------------------ HEADER ------------------

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

# ------------------ SIDEBAR ------------------

st.sidebar.title("Project Information")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Accuracy: 86.89%")
st.sidebar.write("ROC-AUC: 0.9354")
st.sidebar.write("Dataset: Z-Alizadeh Sani")

st.sidebar.markdown("---")
st.sidebar.write("Developer: Tejasvi B")
st.sidebar.write("B.Tech Biotechnology")
st.sidebar.write("BIT, Sathyamangalam")

# ------------------ DESCRIPTION ------------------

st.markdown("""
This application predicts the risk of Coronary Artery Disease (CAD)
using a Random Forest Machine Learning model trained on the
Z-Alizadeh Sani dataset.
""")

# ------------------ INPUT SECTION ------------------

st.markdown("""
<div style="
background-color:#111827;
padding:15px;
border-radius:15px;
border:1px solid #374151;
margin-bottom:15px;
">
<h3 style="color:white;"> Patient Clinical Parameters</h3>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 20, 100, 50)
    bmi = st.number_input("BMI", 10.0, 50.0, 25.0)
    fbs = st.number_input("FBS", 50, 300, 100)
    hdl = st.number_input("HDL", 10, 100, 40)
    sex = st.selectbox("Sex", ["Male", "Female"])
    htn = st.selectbox("Hypertension",  [0, 1])
    

with col2:
    weight = st.number_input("Weight", 30, 150, 70)
    bp = st.number_input("Blood Pressure", 80, 200, 120)
    tg = st.number_input("Triglycerides", 50, 500, 150)
    ldl = st.number_input("LDL", 50, 300, 130)
    dm = st.selectbox("Diabetes", [0, 1])
    
    smoker = st.selectbox("Current Smoker",  [0, 1])
    

chest_pain = st.selectbox("Typical Chest Pain",  [0, 1])
patient_name = st.text_input("Patient Name")
st.info("For Diabetes, Hypertension and Current Smoker: 0 = No, 1 = Yes")

# ------------------ PREDICTION ------------------

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
            'WBC': 7000,
            'Lymph': 22,
            'Neut': 68,
            'PLT': 172,
            'EF-TTE': 60,
            'Region RWMA': 0,
            'Sex_Male': (sex == "Male"),
            'Obesity_Y': False,
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
            'VHD_mild': False
        }])

        prediction = model.predict(patient)[0]
        probability = model.predict_proba(patient)[0][1]

        st.markdown("""
        <div style="
        background-color:#111827;
        padding:15px;
        border-radius:15px;
        border:1px solid #374151;
        margin-top:20px;
        margin-bottom:15px;
        ">
        <h3 style="color:white;">📈 Prediction Results</h3>
        </div>
        """, unsafe_allow_html=True)

        if probability < 0.50:
            risk_text = "LOW"
        elif probability < 0.85:
            risk_text = "MODERATE"
        else:
            risk_text = "HIGH"

        if risk_text == "HIGH":
            st.error("🚨 HIGH CAD RISK DETECTED")
        elif risk_text == "MODERATE":
            st.warning("⚠️ MODERATE CAD RISK DETECTED")
        else:
            st.success("✅ LOW CAD RISK")

        st.write(f"👤 Patient: {patient_name}")

        colA, colB, colC = st.columns(3)

        with colA:
            st.metric("Risk Status", risk_text)

        with colB:
            st.metric("CAD Probability", f"{probability*100:.2f}%")
            
            st.progress(int(probability * 100))

        with colC:
            st.metric("Risk Level", risk_text)
        if risk_text == "LOW":
            st.success("Low probability of Coronary Artery Disease based on the entered parameters.")
        elif risk_text == "MODERATE":
            st.warning("Moderate CAD risk detected. Clinical evaluation is recommended.")
        else:
            st.error("High CAD risk detected. Immediate medical consultation is recommended.")
           
            
        st.success("📋 Patient Summary")
        st.write(f"**Patient Name:** {patient_name}")
        st.write(f"**Age:** {age}")
        st.write(f"**Sex:** {sex}")
        st.write(f"**Diabetes:** {'Yes' if dm == 1 else 'No'}")
        st.write(f"**Hypertension:** {'Yes' if htn == 1 else 'No'}")
        st.write(f"**Current Smoker:** {'Yes' if smoker == 1 else 'No'}")
        st.write(f"**LDL:** {ldl}")
        st.write(f"**BMI:** {bmi}")
        st.write(f"**Blood Pressure:** {bp}")
        st.write(f"**FBS:** {fbs}")
        st.write(f"**HDL:** {hdl}")
        st.write(f"**Triglycerides:** {tg}")

        st.markdown("---")
# ------------------ EXPLAINABILITY ------------------

st.subheader("📊 Model Explainability")
st.image("shap_summary.png")

# ------------------ DISCLAIMER ------------------

st.warning(
    "This tool is for educational and research purposes only and should not be used as a substitute for professional medical advice."
)
st.markdown("---")
st.caption(
    "Developed by Tejasvi B | Nexora | B.Tech Biotechnology | BIT Sathyamangalam | Multimodal AI Hackathon 2026"
)