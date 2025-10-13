"""
API de Inferencia para Taller 5 - Locust
Clasificador de Cobertura Forestal Simplificado
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict
import pickle
import numpy as np
import os
import time
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación FastAPI
app = FastAPI(
    title="Forest Cover Inference API",
    description="API para predicción de cobertura forestal - Taller Locust",
    version="1.0.0"
)

# ==================== MODELOS DE DATOS ====================

class ForestFeatures(BaseModel):
    """Features para predicción de cobertura forestal"""
    Elevation: float = Field(..., description="Elevación en metros")
    Aspect: float = Field(..., description="Aspecto en grados azimuth")
    Slope: float = Field(..., description="Pendiente en grados")
    Horizontal_Distance_To_Hydrology: float = Field(..., description="Distancia horizontal al agua")
    Vertical_Distance_To_Hydrology: float = Field(..., description="Distancia vertical al agua")
    Horizontal_Distance_To_Roadways: float = Field(..., description="Distancia horizontal a caminos")
    Hillshade_9am: float = Field(..., description="Sombra de colina a las 9am")
    Hillshade_Noon: float = Field(..., description="Sombra de colina al mediodía")
    Hillshade_3pm: float = Field(..., description="Sombra de colina a las 3pm")
    Horizontal_Distance_To_Fire_Points: float = Field(..., description="Distancia horizontal a puntos de incendio")
    # Wilderness Area (4 features one-hot encoded)
    Wilderness_Area1: int = Field(0, ge=0, le=1)
    Wilderness_Area2: int = Field(0, ge=0, le=1)
    Wilderness_Area3: int = Field(0, ge=0, le=1)
    Wilderness_Area4: int = Field(0, ge=0, le=1)
    # Soil Type (40 features one-hot encoded)
    Soil_Type1: int = Field(0, ge=0, le=1)
    Soil_Type2: int = Field(0, ge=0, le=1)
    Soil_Type3: int = Field(0, ge=0, le=1)
    Soil_Type4: int = Field(0, ge=0, le=1)
    Soil_Type5: int = Field(0, ge=0, le=1)
    Soil_Type6: int = Field(0, ge=0, le=1)
    Soil_Type7: int = Field(0, ge=0, le=1)
    Soil_Type8: int = Field(0, ge=0, le=1)
    Soil_Type9: int = Field(0, ge=0, le=1)
    Soil_Type10: int = Field(0, ge=0, le=1)
    Soil_Type11: int = Field(0, ge=0, le=1)
    Soil_Type12: int = Field(0, ge=0, le=1)
    Soil_Type13: int = Field(0, ge=0, le=1)
    Soil_Type14: int = Field(0, ge=0, le=1)
    Soil_Type15: int = Field(0, ge=0, le=1)
    Soil_Type16: int = Field(0, ge=0, le=1)
    Soil_Type17: int = Field(0, ge=0, le=1)
    Soil_Type18: int = Field(0, ge=0, le=1)
    Soil_Type19: int = Field(0, ge=0, le=1)
    Soil_Type20: int = Field(0, ge=0, le=1)
    Soil_Type21: int = Field(0, ge=0, le=1)
    Soil_Type22: int = Field(0, ge=0, le=1)
    Soil_Type23: int = Field(0, ge=0, le=1)
    Soil_Type24: int = Field(0, ge=0, le=1)
    Soil_Type25: int = Field(0, ge=0, le=1)
    Soil_Type26: int = Field(0, ge=0, le=1)
    Soil_Type27: int = Field(0, ge=0, le=1)
    Soil_Type28: int = Field(0, ge=0, le=1)
    Soil_Type29: int = Field(0, ge=0, le=1)
    Soil_Type30: int = Field(0, ge=0, le=1)
    Soil_Type31: int = Field(0, ge=0, le=1)
    Soil_Type32: int = Field(0, ge=0, le=1)
    Soil_Type33: int = Field(0, ge=0, le=1)
    Soil_Type34: int = Field(0, ge=0, le=1)
    Soil_Type35: int = Field(0, ge=0, le=1)
    Soil_Type36: int = Field(0, ge=0, le=1)
    Soil_Type37: int = Field(0, ge=0, le=1)
    Soil_Type38: int = Field(0, ge=0, le=1)
    Soil_Type39: int = Field(0, ge=0, le=1)
    Soil_Type40: int = Field(0, ge=0, le=1)

    class Config:
        json_schema_extra = {
            "example": {
                "Elevation": 2596,
                "Aspect": 51,
                "Slope": 3,
                "Horizontal_Distance_To_Hydrology": 258,
                "Vertical_Distance_To_Hydrology": 0,
                "Horizontal_Distance_To_Roadways": 510,
                "Hillshade_9am": 221,
                "Hillshade_Noon": 232,
                "Hillshade_3pm": 148,
                "Horizontal_Distance_To_Fire_Points": 6279,
                "Wilderness_Area1": 1,
                "Wilderness_Area2": 0,
                "Wilderness_Area3": 0,
                "Wilderness_Area4": 0,
                "Soil_Type1": 0,
                "Soil_Type2": 0,
                "Soil_Type3": 0,
                "Soil_Type4": 0,
                "Soil_Type5": 0,
                "Soil_Type6": 0,
                "Soil_Type7": 0,
                "Soil_Type8": 0,
                "Soil_Type9": 0,
                "Soil_Type10": 0,
                "Soil_Type11": 0,
                "Soil_Type12": 0,
                "Soil_Type13": 0,
                "Soil_Type14": 0,
                "Soil_Type15": 0,
                "Soil_Type16": 0,
                "Soil_Type17": 0,
                "Soil_Type18": 0,
                "Soil_Type19": 0,
                "Soil_Type20": 0,
                "Soil_Type21": 0,
                "Soil_Type22": 0,
                "Soil_Type23": 0,
                "Soil_Type24": 0,
                "Soil_Type25": 0,
                "Soil_Type26": 0,
                "Soil_Type27": 0,
                "Soil_Type28": 0,
                "Soil_Type29": 1,
                "Soil_Type30": 0,
                "Soil_Type31": 0,
                "Soil_Type32": 0,
                "Soil_Type33": 0,
                "Soil_Type34": 0,
                "Soil_Type35": 0,
                "Soil_Type36": 0,
                "Soil_Type37": 0,
                "Soil_Type38": 0,
                "Soil_Type39": 0,
                "Soil_Type40": 0
            }
        }

class PredictionResponse(BaseModel):
    """Respuesta de predicción"""
    prediction: int
    cover_type: str
    confidence: float
    processing_time_ms: float
    timestamp: str

# ==================== VARIABLES GLOBALES ====================

# Mapeo de clases a nombres
COVER_TYPES = {
    1: "Spruce/Fir",
    2: "Lodgepole Pine",
    3: "Ponderosa Pine",
    4: "Cottonwood/Willow",
    5: "Aspen",
    6: "Douglas-fir",
    7: "Krummholz"
}

# Modelo cargado
model = None

# Métricas de la API
api_metrics = {
    "total_requests": 0,
    "successful_predictions": 0,
    "failed_predictions": 0,
    "total_processing_time_ms": 0.0,
    "start_time": datetime.now().isoformat()
}

# ==================== FUNCIONES DE CARGA ====================

def load_model():
    """Cargar modelo ML desde disco"""
    global model
    model_path = os.getenv("MODEL_PATH", "/app/models/model.pkl")
    
    try:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"✅ Modelo cargado exitosamente desde {model_path}")
        else:
            # Si no existe el modelo, crear uno dummy para testing
            logger.warning(f"⚠️ Modelo no encontrado en {model_path}, usando modelo dummy")
            from sklearn.ensemble import RandomForestClassifier
            model = RandomForestClassifier(n_estimators=10, random_state=42)
            # Entrenar con datos dummy
            X_dummy = np.random.rand(100, 54)
            y_dummy = np.random.randint(1, 8, 100)
            model.fit(X_dummy, y_dummy)
            logger.info("✅ Modelo dummy creado para testing")
    except Exception as e:
        logger.error(f"❌ Error cargando modelo: {str(e)}")
        raise

# ==================== EVENTOS DE INICIO ====================

@app.on_event("startup")
async def startup_event():
    """Ejecutar al inicio de la aplicación"""
    logger.info("🚀 Iniciando API de Inferencia...")
    load_model()
    logger.info("✅ API lista para recibir peticiones")

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Endpoint raíz - Health check básico"""
    return {
        "message": "Forest Cover Inference API - Taller Locust",
        "status": "running",
        "version": "1.0.0",
        "model_loaded": model is not None,
        "endpoints": {
            "predict": "/predict",
            "health": "/health",
            "metrics": "/metrics",
            "docs": "/docs"
        }
    }

@app.get("/health")
async def health_check():
    """Health check detallado"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "timestamp": datetime.now().isoformat(),
        "uptime_seconds": (datetime.now() - datetime.fromisoformat(api_metrics["start_time"])).total_seconds()
    }

@app.get("/metrics")
async def get_metrics():
    """Obtener métricas de la API"""
    avg_processing_time = (
        api_metrics["total_processing_time_ms"] / api_metrics["total_requests"]
        if api_metrics["total_requests"] > 0 else 0
    )
    
    return {
        "total_requests": api_metrics["total_requests"],
        "successful_predictions": api_metrics["successful_predictions"],
        "failed_predictions": api_metrics["failed_predictions"],
        "success_rate": (
            api_metrics["successful_predictions"] / api_metrics["total_requests"] * 100
            if api_metrics["total_requests"] > 0 else 0
        ),
        "average_processing_time_ms": round(avg_processing_time, 2),
        "start_time": api_metrics["start_time"]
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: ForestFeatures):
    """
    Realizar predicción de cobertura forestal
    
    Args:
        features: Características del área forestal (54 features)
    
    Returns:
        Predicción con tipo de cobertura y confianza
    """
    start_time = time.time()
    api_metrics["total_requests"] += 1
    
    try:
        # Validar que el modelo esté cargado
        if model is None:
            api_metrics["failed_predictions"] += 1
            raise HTTPException(status_code=503, detail="Modelo no disponible")
        
        # Convertir features a array numpy
        feature_dict = features.dict()
        feature_array = np.array([list(feature_dict.values())]).reshape(1, -1)
        
        # Realizar predicción
        prediction = int(model.predict(feature_array)[0])
        
        # Obtener probabilidades si el modelo lo soporta
        try:
            probabilities = model.predict_proba(feature_array)[0]
            confidence = float(max(probabilities))
        except:
            confidence = 0.85  # Confianza por defecto si no hay probabilidades
        
        # Calcular tiempo de procesamiento
        processing_time = (time.time() - start_time) * 1000  # convertir a ms
        
        # Actualizar métricas
        api_metrics["successful_predictions"] += 1
        api_metrics["total_processing_time_ms"] += processing_time
        
        # Preparar respuesta
        response = PredictionResponse(
            prediction=prediction,
            cover_type=COVER_TYPES.get(prediction, "Unknown"),
            confidence=round(confidence, 4),
            processing_time_ms=round(processing_time, 2),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        api_metrics["failed_predictions"] += 1
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")

@app.get("/reset-metrics")
async def reset_metrics():
    """Reiniciar métricas de la API"""
    api_metrics["total_requests"] = 0
    api_metrics["successful_predictions"] = 0
    api_metrics["failed_predictions"] = 0
    api_metrics["total_processing_time_ms"] = 0.0
    api_metrics["start_time"] = datetime.now().isoformat()
    return {"message": "Métricas reiniciadas"}

# ==================== MAIN ====================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8989)
