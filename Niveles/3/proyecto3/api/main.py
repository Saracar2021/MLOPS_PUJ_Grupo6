from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
from datetime import datetime
import pandas as pd
import numpy as np

app = FastAPI(title="Diabetes Readmission Prediction API")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

predictions_counter = Counter('predictions_total', 'Total predictions made', ['model_version', 'prediction_class'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
errors_counter = Counter('prediction_errors_total', 'Total prediction errors')

current_model = None
current_model_version = None

class PredictionRequest(BaseModel):
    race: str = "Caucasian"
    gender: str = "Female"
    age: str = "[50-60)"
    admission_type_id: int = 1
    discharge_disposition_id: int = 1
    admission_source_id: int = 7
    time_in_hospital: int = 3
    payer_code: str = "MC"
    medical_specialty: str = "Cardiology"
    num_lab_procedures: int = 45
    num_procedures: int = 2
    num_medications: int = 15
    number_outpatient: int = 0
    number_emergency: int = 0
    number_inpatient: int = 1
    diag_1: str = "250.83"
    diag_2: str = "401.9"
    diag_3: str = "428.0"
    number_diagnoses: int = 9
    max_glu_serum: str = "None"
    A1Cresult: str = ">8"
    change: str = "yes"
    diabetesMed: str = "yes"

def load_production_model():
    global current_model, current_model_version
    
    try:
        client = MlflowClient()
        model_name = "diabetes_readmission_model"
        
        model_versions = client.get_latest_versions(model_name, stages=["Production"])
        
        if not model_versions:
            raise Exception("No model in Production stage")
        
        latest_version = model_versions[0]
        model_uri = f"models:/{model_name}/Production"
        
        current_model = mlflow.sklearn.load_model(model_uri)
        current_model_version = f"{model_name}_v{latest_version.version}"
        
        print(f"Loaded model: {current_model_version}")
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.on_event("startup")
async def startup_event():
    load_production_model()

@app.get("/")
def read_root():
    return {
        "service": "Diabetes Readmission Prediction API",
        "status": "running",
        "model_version": current_model_version,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    if current_model is None:
        return {"status": "unhealthy", "reason": "Model not loaded"}
    return {"status": "healthy", "model_version": current_model_version}

@app.post("/predict")
@prediction_latency.time()
def predict(request: PredictionRequest):
    if current_model is None:
        errors_counter.inc()
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        features = preprocess_input(request.dict())
        
        prediction = current_model.predict([features])[0]
        probabilities = current_model.predict_proba([features])[0]
        
        class_map = {0: "NO", 1: ">30", 2: "<30"}
        prediction_class = class_map[prediction]
        
        predictions_counter.labels(
            model_version=current_model_version,
            prediction_class=prediction_class
        ).inc()
        
        return {
            "prediction": prediction_class,
            "probabilities": {
                "NO": float(probabilities[0]),
                ">30": float(probabilities[1]),
                "<30": float(probabilities[2])
            },
            "model_version": current_model_version,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        errors_counter.inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
def reload_model():
    success = load_production_model()
    if success:
        return {"status": "success", "model_version": current_model_version}
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

def preprocess_input(data):
    import re
    
    def age_to_numeric(age_str):
        match = re.search(r'\[(\d+)-(\d+)\)', age_str)
        if match:
            low = int(match.group(1))
            high = int(match.group(2))
            return (low + high) / 2
        return 55
    
    def group_icd9(code):
        if not code or code == '?':
            return 'unknown'
        code_str = str(code)
        if code_str.startswith('250'):
            return 'diabetes'
        elif code_str.startswith(('390', '401', '410', '428')):
            return 'circulatory'
        elif code_str.startswith(('460', '480', '490')):
            return 'respiratory'
        elif code_str.startswith(('520', '530', '550')):
            return 'digestive'
        elif code_str.startswith(('580', '600')):
            return 'genitourinary'
        elif code_str.startswith(('710', '720')):
            return 'musculoskeletal'
        elif code_str.startswith(('800', '850')):
            return 'injury'
        else:
            return 'other'
    
    features = {}
    
    features['age_numeric'] = age_to_numeric(data['age'])
    features['race'] = data['race'] if data['race'] != '?' else 'Other/Unknown'
    features['gender'] = data['gender']
    features['payer_code'] = data['payer_code'] if data['payer_code'] != '?' else 'Unknown'
    features['medical_specialty'] = data['medical_specialty'] if data['medical_specialty'] != '?' else 'Unknown'
    
    features['diag_1_grouped'] = group_icd9(data['diag_1'])
    features['diag_2_grouped'] = group_icd9(data['diag_2'])
    features['diag_3_grouped'] = group_icd9(data['diag_3'])
    
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    features['max_glu_serum'] = glu_map.get(data['max_glu_serum'], 0)
    
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    features['A1Cresult'] = a1c_map.get(data['A1Cresult'], 0)
    
    features['change'] = 1 if data['change'].lower() in ['ch', 'yes'] else 0
    features['diabetesMed'] = 1 if data['diabetesMed'].lower() == 'yes' else 0
    
    for key in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']:
        features[key] = data[key]
    
    medication_columns = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
        'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
        'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
        'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
        'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    
    for med in medication_columns:
        features[med] = 0
    
    feature_order = [
        'age_numeric', 'race', 'gender', 'payer_code', 'medical_specialty',
        'admission_type_id', 'discharge_disposition_id', 'admission_source_id',
        'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
        'number_outpatient', 'number_emergency', 'number_inpatient',
        'diag_1_grouped', 'diag_2_grouped', 'diag_3_grouped', 'number_diagnoses',
        'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed'
    ] + medication_columns
    
    return [features.get(f, 0) for f in feature_order]

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
