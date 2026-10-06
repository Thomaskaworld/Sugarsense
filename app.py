import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ==========================================================
# SugarSense - Early Diabetes Risk Screening
# ==========================================================

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------

st.set_page_config(
    page_title="SugarSense",
    page_icon="🩺",
    layout="centered"
)

# ----------------------------------------------------------
# Custom Styling
# ----------------------------------------------------------

st.markdown("""
<style>

    .main-title {
        text-align: center;
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 0px;
    }

    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #888888;
        margin-bottom: 30px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 600;
        margin-top: 15px;
        margin-bottom: 10px;
    }

    .info-box {
        padding: 15px;
        border-radius: 10px;
        background-color: rgba(49, 51, 63, 0.5);
        border: 1px solid rgba(128, 128, 128, 0.2);
        margin-bottom: 20px;
    }

    .result-title {
        text-align: center;
        font-size: 28px;
        font-weight: 600;
    }

    .footer {
        text-align: center;
        color: #888888;
        font-size: 13px;
        margin-top: 30px;
    }

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load Model and Scaler
# ----------------------------------------------------------

model = joblib.load("sugarsense_model.pkl")
scaler = joblib.load("sugarsense_scaler.pkl")

# Tuned threshold obtained during model evaluation
BEST_THRESHOLD = 0.21

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------

st.markdown(
    '<div class="main-title">🩺 SugarSense</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Early Diabetes Risk Screening using Machine Learning</div>',
    unsafe_allow_html=True
)

st.markdown("""
<div class="info-box">

<b>About SugarSense</b><br><br>

SugarSense uses a supervised machine learning model to estimate
diabetes risk from commonly used clinical measurements.

Enter the patient's information below to generate a screening result.

</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Patient Information
# ----------------------------------------------------------

st.markdown(
    '<div class="section-title">👤 Patient Information</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1,
        step=1
    )

    glucose = st.number_input(
        "Glucose (mg/dL)",
        min_value=0.0,
        max_value=250.0,
        value=120.0,
        step=1.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure (mm Hg)",
        min_value=0.0,
        max_value=150.0,
        value=70.0,
        step=1.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness (mm)",
        min_value=0.0,
        max_value=100.0,
        value=25.0,
        step=1.0
    )

with col2:

    insulin = st.number_input(
        "Insulin (µU/mL)",
        min_value=0.0,
        max_value=900.0,
        value=80.0,
        step=1.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        value=30.0,
        step=0.1
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.50,
        step=0.01
    )

    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=30,
        step=1
    )

# ----------------------------------------------------------
# Prediction Button
# ----------------------------------------------------------

st.divider()

check_risk = st.button(
    "🔍 Check Diabetes Risk",
    use_container_width=True
)

# ----------------------------------------------------------
# Prediction
# ----------------------------------------------------------

if check_risk:

    # Keep the exact feature order used during model training
    patient_data = pd.DataFrame(
        [[
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree,
            age
        ]],
        columns=[
            "Pregnancies",
            "Glucose",
            "BloodPressure",
            "SkinThickness",
            "Insulin",
            "BMI",
            "DiabetesPedigreeFunction",
            "Age"
        ]
    )

    # Apply the same scaler used during training
    patient_scaled = scaler.transform(patient_data)

    # Get diabetes probability
    probability = model.predict_proba(patient_scaled)[0][1]

    # Apply tuned threshold
    prediction = int(probability >= BEST_THRESHOLD)

    # ------------------------------------------------------
    # Result
    # ------------------------------------------------------

    st.markdown(
        '<div class="result-title">📊 Screening Result</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk"
        )

    else:

        st.success(
            "✅ Lower Diabetes Risk"
        )

    # Probability
    st.metric(
        label="Estimated Risk Probability",
        value=f"{probability:.2%}"
    )

    st.progress(float(probability))

    # ------------------------------------------------------
    # Interpretation
    # ------------------------------------------------------

    if probability < 0.21:

        st.info(
            "The model estimates a lower diabetes risk based on "
            "the entered measurements."
        )

    else:

        st.warning(
            "The model estimates a higher diabetes risk based on "
            "the entered measurements."
        )

    # ------------------------------------------------------
    # Important Disclaimer
    # ------------------------------------------------------

    st.warning(
        "⚕️ This application is intended for educational and "
        "screening purposes only. The result is generated by "
        "a machine learning model and should not be considered "
        "a medical diagnosis."
    )

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.divider()

st.markdown(
    """
    <div class="footer">
        SugarSense | Early Diabetes Risk Screening with
        Supervised Machine Learning<br>
        Logistic Regression • Threshold Tuning • Model Interpretation
    </div>
    """,
    unsafe_allow_html=True
)