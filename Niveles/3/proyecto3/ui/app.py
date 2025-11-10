import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(
    page_title="Diabetes Readmission Predictor",
    page_icon="🏥",
    layout="wide"
)

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")

st.title("🏥 Hospital Readmission Prediction System")
st.markdown("### Diabetes Patients - 30-Day Readmission Risk")

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Patient Information")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        race = st.selectbox("Race", ["Caucasian", "AfricanAmerican", "Hispanic", "Asian", "Other/Unknown"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.selectbox("Age Range", ["[0-10)", "[10-20)", "[20-30)", "[30-40)", "[40-50)", 
                                         "[50-60)", "[60-70)", "[70-80)", "[80-90)", "[90-100)"])
    
    with col_b:
        admission_type = st.number_input("Admission Type ID", min_value=1, max_value=8, value=1)
        discharge_disp = st.number_input("Discharge Disposition ID", min_value=1, max_value=30, value=1)
        admission_source = st.number_input("Admission Source ID", min_value=1, max_value=25, value=7)
    
    st.subheader("Clinical Data")
    
    col_c, col_d, col_e = st.columns(3)
    
    with col_c:
        time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, max_value=14, value=3)
        num_lab_procedures = st.number_input("Lab Procedures", min_value=0, max_value=100, value=45)
        num_procedures = st.number_input("Procedures", min_value=0, max_value=10, value=2)
    
    with col_d:
        num_medications = st.number_input("Medications", min_value=1, max_value=50, value=15)
        number_outpatient = st.number_input("Outpatient Visits", min_value=0, max_value=20, value=0)
        number_emergency = st.number_input("Emergency Visits", min_value=0, max_value=20, value=0)
    
    with col_e:
        number_inpatient = st.number_input("Inpatient Visits", min_value=0, max_value=20, value=1)
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=1, max_value=16, value=9)
    
    st.subheader("Medical History")
    
    col_f, col_g = st.columns(2)
    
    with col_f:
        payer_code = st.text_input("Payer Code", value="MC")
        medical_specialty = st.text_input("Medical Specialty", value="Cardiology")
        diag_1 = st.text_input("Primary Diagnosis (ICD-9)", value="250.83")
    
    with col_g:
        diag_2 = st.text_input("Secondary Diagnosis", value="401.9")
        diag_3 = st.text_input("Tertiary Diagnosis", value="428.0")
    
    st.subheader("Diabetes Management")
    
    col_h, col_i = st.columns(2)
    
    with col_h:
        max_glu_serum = st.selectbox("Max Glucose Serum", ["None", "Norm", ">200", ">300"])
        A1Cresult = st.selectbox("A1C Result", ["None", "Norm", ">7", ">8"])
    
    with col_i:
        change = st.selectbox("Medication Change", ["no", "yes"])
        diabetesMed = st.selectbox("Diabetes Medication", ["no", "yes"])

with col2:
    st.subheader("Model Information")
    
    try:
        health_response = requests.get(f"{FASTAPI_URL}/health")
        if health_response.status_code == 200:
            health_data = health_response.json()
            st.success("✅ API Status: Healthy")
            st.info(f"**Model Version:** {health_data.get('model_version', 'Unknown')}")
        else:
            st.error("❌ API Status: Unhealthy")
    except:
        st.error("❌ Cannot connect to API")
    
    st.markdown("---")
    
    if st.button("🔄 Reload Model", help="Fetch latest production model from MLflow"):
        try:
            reload_response = requests.post(f"{FASTAPI_URL}/reload_model")
            if reload_response.status_code == 200:
                data = reload_response.json()
                st.success(f"Model reloaded: {data['model_version']}")
            else:
                st.error("Failed to reload model")
        except:
            st.error("Cannot connect to API")

st.markdown("---")

col_predict, col_preset = st.columns([3, 1])

with col_predict:
    if st.button("🔍 Predict Readmission Risk", type="primary", use_container_width=True):
        payload = {
            "race": race,
            "gender": gender,
            "age": age,
            "admission_type_id": admission_type,
            "discharge_disposition_id": discharge_disp,
            "admission_source_id": admission_source,
            "time_in_hospital": time_in_hospital,
            "payer_code": payer_code,
            "medical_specialty": medical_specialty,
            "num_lab_procedures": num_lab_procedures,
            "num_procedures": num_procedures,
            "num_medications": num_medications,
            "number_outpatient": number_outpatient,
            "number_emergency": number_emergency,
            "number_inpatient": number_inpatient,
            "diag_1": diag_1,
            "diag_2": diag_2,
            "diag_3": diag_3,
            "number_diagnoses": number_diagnoses,
            "max_glu_serum": max_glu_serum,
            "A1Cresult": A1Cresult,
            "change": change,
            "diabetesMed": diabetesMed
        }
        
        try:
            response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("### Prediction Complete!")
                
                prediction = result['prediction']
                probabilities = result['probabilities']
                model_version = result['model_version']
                
                risk_color = {
                    "<30": "🔴",
                    ">30": "🟡",
                    "NO": "🟢"
                }
                
                risk_message = {
                    "<30": "HIGH RISK - Readmission within 30 days",
                    ">30": "MEDIUM RISK - Readmission after 30 days",
                    "NO": "LOW RISK - No readmission expected"
                }
                
                st.markdown(f"## {risk_color[prediction]} {risk_message[prediction]}")
                
                st.markdown("### Probability Distribution")
                prob_df = pd.DataFrame({
                    'Class': ['No Readmission', 'Readmit >30 days', 'Readmit <30 days'],
                    'Probability': [probabilities['NO'], probabilities['>30'], probabilities['<30']]
                })
                st.bar_chart(prob_df.set_index('Class'))
                
                st.markdown(f"**Model Used:** `{model_version}`")
                st.markdown(f"**Timestamp:** {result['timestamp']}")
            else:
                st.error(f"Prediction failed: {response.text}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

with col_preset:
    if st.button("📝 Load Example", use_container_width=True):
        st.info("Example values loaded! Click Predict to see results.")

st.markdown("---")
st.caption("MLOps Proyecto 3 - Grupo 6 - Pontificia Universidad Javeriana")
