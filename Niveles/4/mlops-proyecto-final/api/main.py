from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Real Estate Price Prediction API", version="1.0.0")

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
MODEL_NAME = "real_estate_model"

# Métricas Prometheus
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

def load_production_model():
    """Cargar modelo desde MLflow Production stage"""
    global current_model, current_model_version, current_model_rmse
    
    try:
        mlflow.set_tracking_uri(MLFLOW_URI)
        client = MlflowClient()
        
        # Obtener última versión en Production
        versions = client.get_latest_versions(MODEL_NAME, stages=["Production"])
        
        if not versions:
            logger.error("❌ No hay modelo en Production")
            return False
        
        latest = versions[0]
        model_uri = f"models:/{MODEL_NAME}/Production"
        
        current_model = mlflow.sklearn.load_model(model_uri)
        current_model_version = f"v{latest.version}"
        
        # Obtener RMSE del run
        run = client.get_run(latest.run_id)
        current_model_rmse = run.data.metrics.get('rmse', 0)
        model_rmse_gauge.set(current_model_rmse)
        
        logger.info(f"✅ Modelo cargado: {MODEL_NAME} {current_model_version}")
        logger.info(f"   RMSE: {current_model_rmse:.2f}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
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
        # Preparar features (simplificado - en producción haría encoding completo)
        features = pd.DataFrame([{
            'brokered_by': request.brokered_by,
            'status': 1 if request.status == 'for_sale' else 0,
            'bed': request.bed,
            'bath': request.bath,
            'acre_lot': request.acre_lot,
            'city': request.city,
            'state': request.state,
            'zip_code': request.zip_code,
            'house_size': request.house_size
        }])
        
        # Predicción
        prediction = current_model.predict(features)[0]
        
        # Métricas
        predictions_counter.labels(model_version=current_model_version).inc()
        
        logger.info(f"✅ Predicción: ${prediction:,.2f} (modelo: {current_model_version})")
        
        return {
            "predicted_price": float(prediction),
            "model_version": current_model_version,
            "model_rmse": current_model_rmse,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        errors_counter.inc()
        logger.error(f"❌ Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/reload_model")
def reload_model():
    """Recargar modelo desde MLflow"""
    logger.info("🔄 Recargando modelo desde MLflow...")
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
