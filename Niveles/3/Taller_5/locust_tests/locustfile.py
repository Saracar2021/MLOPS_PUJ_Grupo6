"""
Locustfile para pruebas de carga de la API de Inferencia
Taller 5 - Locust - MLOps

Simula usuarios realizando predicciones de cobertura forestal
"""

from locust import HttpUser, task, between, events
import random
import json
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ====================================================================
# DATOS DE PRUEBA
# ====================================================================

# Generar datos sintéticos para las pruebas
def generate_random_features():
    """Generar features aleatorias para una predicción"""
    
    # Features base (10 numéricas)
    features = {
        "Elevation": random.uniform(1500, 4000),
        "Aspect": random.uniform(0, 360),
        "Slope": random.uniform(0, 50),
        "Horizontal_Distance_To_Hydrology": random.uniform(0, 1000),
        "Vertical_Distance_To_Hydrology": random.uniform(-200, 200),
        "Horizontal_Distance_To_Roadways": random.uniform(0, 5000),
        "Hillshade_9am": random.uniform(0, 255),
        "Hillshade_Noon": random.uniform(0, 255),
        "Hillshade_3pm": random.uniform(0, 255),
        "Horizontal_Distance_To_Fire_Points": random.uniform(0, 7000)
    }
    
    # Wilderness Area (4 features one-hot encoded)
    wilderness = random.randint(1, 4)
    for i in range(1, 5):
        features[f"Wilderness_Area{i}"] = 1 if i == wilderness else 0
    
    # Soil Type (40 features one-hot encoded)
    soil_type = random.randint(1, 40)
    for i in range(1, 41):
        features[f"Soil_Type{i}"] = 1 if i == soil_type else 0
    
    return features

# Ejemplos pre-generados para mayor velocidad
SAMPLE_FEATURES = [generate_random_features() for _ in range(100)]

# ====================================================================
# USUARIO SIMULADO
# ====================================================================

class ForestCoverUser(HttpUser):
    """
    Usuario simulado que realiza predicciones de cobertura forestal
    """
    
    # Tiempo de espera entre peticiones (en segundos)
    wait_time = between(0.1, 0.5)  # Entre 100ms y 500ms
    
    # Host de la API (se sobrescribe en la UI de Locust)
    host = "http://localhost:8989"
    
    def on_start(self):
        """Ejecutar una vez cuando el usuario inicia"""
        logger.info("🚀 Nuevo usuario iniciado")
        
        # Verificar que la API está disponible
        try:
            response = self.client.get("/health", name="00_Health_Check")
            if response.status_code == 200:
                logger.info("✅ API disponible")
            else:
                logger.warning(f"⚠️ API respondió con código {response.status_code}")
        except Exception as e:
            logger.error(f"❌ Error conectando con API: {str(e)}")
    
    @task(10)  # Peso 10 - tarea más común
    def predict_forest_cover(self):
        """
        Tarea: Realizar predicción de cobertura forestal
        Esta es la tarea principal y más frecuente
        """
        # Seleccionar features aleatorias
        features = random.choice(SAMPLE_FEATURES)
        
        # Realizar petición POST
        with self.client.post(
            "/predict",
            json=features,
            name="01_Predict",
            catch_response=True
        ) as response:
            
            if response.status_code == 200:
                try:
                    result = response.json()
                    # Validar respuesta
                    if "prediction" in result and "cover_type" in result:
                        response.success()
                    else:
                        response.failure("Respuesta inválida: faltan campos")
                except json.JSONDecodeError:
                    response.failure("Error al decodificar JSON")
            else:
                response.failure(f"Código de estado: {response.status_code}")
    
    @task(2)  # Peso 2 - menos frecuente
    def check_health(self):
        """
        Tarea: Verificar salud de la API
        """
        with self.client.get(
            "/health",
            name="02_Health_Check",
            catch_response=True
        ) as response:
            
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check falló: {response.status_code}")
    
    @task(1)  # Peso 1 - ocasional
    def get_metrics(self):
        """
        Tarea: Obtener métricas de la API
        """
        with self.client.get(
            "/metrics",
            name="03_Get_Metrics",
            catch_response=True
        ) as response:
            
            if response.status_code == 200:
                try:
                    metrics = response.json()
                    # Log de métricas interesantes
                    if metrics.get("total_requests", 0) % 100 == 0:
                        logger.info(f"📊 Métricas API: {metrics.get('total_requests')} requests, "
                                  f"{metrics.get('success_rate', 0):.2f}% success rate")
                    response.success()
                except:
                    response.failure("Error al obtener métricas")
            else:
                response.failure(f"Métricas no disponibles: {response.status_code}")

# ====================================================================
# EVENTOS DE LOCUST
# ====================================================================

@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Ejecutar cuando la prueba de carga inicia"""
    logger.info("=" * 60)
    logger.info("🚀 INICIANDO PRUEBAS DE CARGA")
    logger.info("=" * 60)
    logger.info(f"Host: {environment.host}")
    logger.info(f"Usuarios: {environment.parsed_options.num_users if hasattr(environment, 'parsed_options') else 'N/A'}")
    logger.info("=" * 60)

@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Ejecutar cuando la prueba de carga termina"""
    logger.info("=" * 60)
    logger.info("🏁 PRUEBAS DE CARGA FINALIZADAS")
    logger.info("=" * 60)
    
    # Obtener estadísticas
    stats = environment.stats
    logger.info(f"Total de peticiones: {stats.total.num_requests}")
    logger.info(f"Peticiones fallidas: {stats.total.num_failures}")
    logger.info(f"RPS promedio: {stats.total.total_rps:.2f}")
    logger.info(f"Tiempo de respuesta (mediana): {stats.total.median_response_time}ms")
    logger.info(f"Tiempo de respuesta (P95): {stats.total.get_response_time_percentile(0.95)}ms")
    logger.info("=" * 60)

@events.request.add_listener
def on_request(request_type, name, response_time, response_length, exception, **kwargs):
    """Ejecutar en cada petición (útil para logging detallado)"""
    # Solo loguear si hay error
    if exception:
        logger.error(f"❌ Error en {name}: {exception}")

# ====================================================================
# CLASES ADICIONALES PARA DIFERENTES ESCENARIOS
# ====================================================================

class LightUser(HttpUser):
    """
    Usuario ligero - Solo health checks
    Útil para pruebas de disponibilidad
    """
    wait_time = between(1, 3)
    host = "http://localhost:8989"
    
    @task
    def health_check(self):
        self.client.get("/health", name="Light_Health_Check")

class HeavyUser(HttpUser):
    """
    Usuario pesado - Múltiples predicciones sin espera
    Útil para pruebas de estrés
    """
    wait_time = between(0.01, 0.05)  # Muy poco tiempo de espera
    host = "http://localhost:8989"
    
    @task
    def rapid_predictions(self):
        features = random.choice(SAMPLE_FEATURES)
        self.client.post("/predict", json=features, name="Heavy_Predict")

# ====================================================================
# INSTRUCCIONES DE USO
# ====================================================================
"""
EJECUTAR LOCUST:

1. Modo Web UI:
   locust -f locustfile.py --host=http://localhost:8989
   
   Luego abre: http://localhost:8089
   Configura:
   - Number of users: 10000
   - Spawn rate: 500 (usuarios por segundo)
   - Host: http://localhost:8989

2. Modo Headless (sin UI):
   locust -f locustfile.py --host=http://localhost:8989 \
          --users 10000 --spawn-rate 500 --run-time 5m

3. Con usuarios específicos:
   locust -f locustfile.py --host=http://localhost:8989 \
          ForestCoverUser

4. Modo distribuido (master + workers):
   # En terminal 1 (master):
   locust -f locustfile.py --master
   
   # En terminal 2,3,4... (workers):
   locust -f locustfile.py --worker --master-host=localhost

ANÁLISIS DE RESULTADOS:
- RPS (Requests per Second): Objetivo >500
- Response Time P95: Objetivo <500ms
- Error Rate: Objetivo <1%
- CPU Usage: Monitorear con 'docker stats'
"""
