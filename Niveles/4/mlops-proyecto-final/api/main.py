from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow
import mlflow.pyfunc
from mlflow.tracking import MlflowClient
import os
import logging
from datetime import datetime
import pandas as pd
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Price Prediction API", version="1.0.0")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "real_estate_model"

# Métricas
predictions_counter = Counter('predictions_total', 'Total predictions', ['model_version'])
prediction_latency = Histogram('prediction_latency_seconds', 'Prediction latency')
errors_counter = Counter('prediction_errors_total', 'Prediction errors')
model_rmse_gauge = Gauge('model_rmse', 'Current model RMSE')

# Estado global
current_model = None
current_model_version = None
current_model_rmse = None

class PredictionRequest(BaseModel):
    brokered_by: str
    status: str
    bed: int
    bath: float
    acre_lot: float
    city: str
    state: str
    zip_code: str
    house_size: float

def preprocess_input(data):
    """Convierte texto a números usando Hash Encoding para evitar error de tipos"""
    features = {}
    
    # 1. Numéricos: Asegurar tipo
    features['bed'] = int(data.get('bed', 0))
    features['bath'] = float(data.get('bath', 0))
    features['acre_lot'] = float(data.get('acre_lot', 0))
    features['house_size'] = float(data.get('house_size', 0))
    
    # 2. Categóricos: Convertir string -> int (Hash)
    categorical_cols = ['brokered_by', 'status', 'city', 'state', 'zip_code']
    for col in categorical_cols:
        value = str(data.get(col, 'unknown'))
        # Generar un entero determinista basado en el texto
        hash_val = int(hashlib.md5(value.encode()).hexdigest(), 16)
        features[col] = int(hash_val % 10000)  # Reducir a un rango manejable
    
    return features

def load_production_model():
    global current_model, current_model_version, current_model_rmse
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        # Buscar en stage Production o el último run exitoso
        try:
            versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if versions:
                latest = versions[0]
                model_uri = f"models:/{MODEL_NAME}/Production"
                version_label = f"v{latest.version} (Prod)"
                run_id = latest.run_id
            else:
                raise Exception("No production model")
        except:
            # Fallback: Usar el último run exitoso
            experiment = client.get_experiment_by_name("real_estate_prediction")
            runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
            if not runs: return False
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/model"
            version_label = f"Run {run_id[:7]}"
            
        current_model = mlflow.pyfunc.load_model(model_uri)
        current_model_version = version_label
        
        # Intentar obtener métrica RMSE
        run = client.get_run(run_id)
        current_model_rmse = run.data.metrics.get('rmse', 0)
        model_rmse_gauge.set(current_model_rmse)
        
        logger.info(f"✅ Modelo cargado: {version_label}")
        return True
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {e}")
        return False

@app.on_event("startup")
async def startup_event():
    load_production_model()

@app.post("/predict")
@prediction_latency.time()
def predict(request: PredictionRequest):
    if current_model is None:
        load_production_model()
        if current_model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # AQUÍ OCURRE LA MAGIA: Preprocesar antes de crear el DataFrame
        features_dict = preprocess_input(request.dict())
        
        # Crear DF con tipos explícitos
        df = pd.DataFrame([features_dict])
        
        # Ordenar columnas como espera el modelo (según training)
        feature_cols = ['bed', 'bath', 'acre_lot', 'house_size', 
                       'brokered_by', 'status', 'city', 'state', 'zip_code']
        
        # Asegurar que todas las columnas existan
        for col in feature_cols:
            if col not in df.columns:
                df[col] = 0
                
        df = df[feature_cols] # Reordenar
        
        prediction = current_model.predict(df)[0]
        
        predictions_counter.labels(model_version=str(current_model_version)).inc()
        
        return {
            "predicted_price": float(prediction),
            "model_version": current_model_version,
            "model_rmse": current_model_rmse
        }
    
    except Exception as e:
        errors_counter.inc()
        logger.error(f"Error predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
def health():
    return {"status": "healthy" if current_model else "loading", "model_version": current_model_version}

@app.post("/reload_model")
def reload():
    if load_production_model():
        return {"status": "reloaded", "version": current_model_version}
    raise HTTPException(status_code=500, detail="Failed to reload")
