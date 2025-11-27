from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow
import mlflow.pyfunc  # CAMBIO: Universal para todos los modelos
from mlflow.tracking import MlflowClient
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Price Prediction API", version="1.0.0")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "real_estate_model"

# Métricas Prometheus
predictions_counter = Counter('predictions_total', 'Total predictions', ['model_version'])  # 1 label
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
    """Preprocess input data - returns dict with native Python types"""
    features = {}
    
    # Numeric features
    features['bed'] = int(data.get('bed', 0))
    features['bath'] = float(data.get('bath', 0))
    features['acre_lot'] = float(data.get('acre_lot', 0))
    features['house_size'] = float(data.get('house_size', 0))
    
    # Categorical features - hash encoding
    categorical_cols = ['brokered_by', 'status', 'city', 'state', 'zip_code']
    for col in categorical_cols:
        value = str(data.get(col, 'unknown'))
        hash_val = int(hashlib.md5(value.encode()).hexdigest(), 16)
        features[col] = int(hash_val % 10000)
    
    return features

def load_production_model():
    """Cargar modelo desde MLflow Production stage"""
    global current_model, current_model_version, current_model_rmse
    
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if not versions:
            logger.error("❌ No hay modelo en Production")
            return False
        
        latest = versions[0]
        model_uri = f"models:/{MODEL_NAME}/Production"
        
        # CAMBIO: Usar pyfunc (funciona con sklearn, xgboost, etc)
        current_model = mlflow.pyfunc.load_model(model_uri)
        current_model_version = f"v{latest.version}"
        
        # Obtener RMSE
        run = client.get_run(latest.run_id)
        current_model_rmse = run.data.metrics.get('rmse', 0)
        model_rmse_gauge.set(current_model_rmse)
        
        logger.info(f"✅ Modelo cargado: {MODEL_NAME} {current_model_version}")
        logger.info(f"   RMSE: {current_model_rmse:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

@app.on_event("startup")
async def startup_event():
    """Cargar modelo al iniciar"""
    logger.info("🚀 Iniciando API...")
    load_production_model()

@app.get("/")
def read_root():
    """Healthcheck básico"""
    return {
        "service": "Real Estate Price Prediction API",
        "status": "running",
        "model_name": MODEL_NAME,
        "model_version": current_model_version,
        "model_rmse": current_model_rmse,
        "mlflow_uri": MLFLOW_URI,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/health")
def health_check():
    """Health check detallado"""
    if current_model is None:
        return {
            "status": "unhealthy",
            "reason": "Model not loaded",
            "mlflow_uri": MLFLOW_URI
        }
    return {
        "status": "healthy",
        "model_version": current_model_version,
        "model_rmse": current_model_rmse
    }

@app.post("/predict")
@prediction_latency.time()
def predict(request: PredictionRequest):
    """Realizar predicción"""
    if current_model is None:
        errors_counter.inc()
        raise HTTPException(status_code=503, detail="Model not available")
    
    try:
        # Preprocess
        features_dict = preprocess_input(request.dict())
        
        # DEBUG
        print(f"🔍 Features dict: {features_dict}")
        print(f"🔍 Types: {[(k, type(v)) for k, v in features_dict.items()]}")
        
        # Create DataFrame
        df = pd.DataFrame([features_dict])
        
        # Explicit dtypes
        dtype_map = {
            'bed': 'int64',
            'bath': 'float64',
            'acre_lot': 'float64',
            'house_size': 'float64',
            'brokered_by': 'int64',
            'status': 'int64',
            'city': 'int64',
            'state': 'int64',
            'zip_code': 'int64'
        }
        
        for col, dtype in dtype_map.items():
            df[col] = df[col].astype(dtype)
        
        # Column order
        feature_cols = ['bed', 'bath', 'acre_lot', 'house_size', 
                       'brokered_by', 'status', 'city', 'state', 'zip_code']
        df = df[feature_cols]
        
        # DEBUG
        print(f"🔍 DataFrame dtypes:\n{df.dtypes}")
        print(f"🔍 DataFrame shape: {df.shape}")
        print(f"🔍 DataFrame values:\n{df.values}")
        
        # Predict
        prediction = current_model.predict(df)[0]
        
        # Metrics (SOLO 1 label)
        predictions_counter.labels(model_version=current_model_version).inc()
        
        return {
            "predicted_price": float(prediction),
            "model_version": current_model_version,
            "model_rmse": current_model_rmse,
            "timestamp": datetime.now().isoformat()
        }
    
    except Exception as e:
        errors_counter.inc()
        print(f"❌ Error en predicción: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
def reload_model():
    """Recargar modelo desde MLflow"""
    logger.info("🔄 Recargando modelo...")
    success = load_production_model()
    if success:
        return {
            "status": "success",
            "model_version": current_model_version,
            "model_rmse": current_model_rmse
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/metrics")
def metrics():
    """Endpoint para Prometheus"""
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
