# 🎯 PROYECTO 3 - RESUMEN EJECUTIVO

## ¿QUÉ ES ESTE PROYECTO?

Sistema completo de MLOps para predecir readmisión hospitalaria de pacientes diabéticos, implementado con Kubernetes, Airflow, MLflow, Prometheus y Grafana.

---

## 📦 CONTENIDO DEL PROYECTO

```
proyecto3_mlops/
├── README.md                          # Instrucciones principales
├── deploy.sh                          # Script de despliegue automatizado
│
├── k8s/                               # Manifiestos Kubernetes
│   ├── namespace.yaml                 # Namespace mlops-proyecto3
│   ├── postgres-raw/                  # DB para datos sin procesar
│   ├── postgres-clean/                # DB para datos procesados
│   ├── postgres-mlflow/               # DB para metadatos MLflow
│   ├── minio/                         # Object storage (S3-compatible)
│   ├── mlflow/                        # Servidor MLflow
│   ├── airflow/                       # Orquestador de pipelines
│   ├── fastapi/                       # API de inferencia
│   ├── streamlit/                     # Interfaz gráfica
│   ├── prometheus/                    # Recolección de métricas
│   └── grafana/                       # Visualización de métricas
│
├── airflow/
│   └── dags/
│       ├── data_ingestion_batch.py    # Carga datos en batches 15k
│       ├── data_processing.py         # RAW → CLEAN
│       ├── model_training.py          # Entrena 3 modelos
│       └── model_promotion.py         # Promociona mejor a Production
│
├── api/
│   └── main.py                        # FastAPI + Prometheus metrics
│
├── ui/
│   └── app.py                         # Streamlit para predicciones
│
├── locust/
│   └── locustfile.py                  # Pruebas de carga
│
└── docs/
    ├── EVIDENCIAS.md                  # Guía de validación
    ├── QUICK_REFERENCE.md             # Comandos rápidos
    └── WINDOWS_GITBASH.md             # Instrucciones Windows
```

---

## 🚀 INICIO RÁPIDO (5 PASOS)

### 1. Requisitos Previos
- Docker Desktop instalado y corriendo
- Minikube instalado
- 32GB RAM (asignar 16GB a Minikube)
- Git Bash (Windows)

### 2. Descomprimir Proyecto
```bash
# Descomprimir proyecto3_mlops.zip
cd proyecto3_mlops
```

### 3. Iniciar Minikube
```bash
minikube start --memory=16384 --cpus=6 --driver=docker
```

### 4. Desplegar Sistema
```bash
# Opción A: Automatizado
chmod +x deploy.sh
./deploy.sh

# Opción B: Manual
kubectl create namespace mlops-proyecto3
kubectl config set-context --current --namespace=mlops-proyecto3
kubectl apply -f k8s/postgres-raw/
kubectl apply -f k8s/postgres-clean/
kubectl apply -f k8s/postgres-mlflow/
kubectl apply -f k8s/minio/
kubectl apply -f k8s/mlflow/
kubectl apply -f k8s/airflow/
kubectl apply -f k8s/fastapi/
kubectl apply -f k8s/streamlit/
kubectl apply -f k8s/prometheus/
kubectl apply -f k8s/grafana/
```

### 5. Acceder a Servicios
```bash
# Ver todas las URLs
minikube service list -n mlops-proyecto3

# Streamlit (interfaz principal)
minikube service streamlit -n mlops-proyecto3
```

---

## 🎬 FLUJO DEL SISTEMA

```
1. Dataset (UCI ML Repo)
   ↓
2. Airflow DAG: data_ingestion_batch
   ↓
3. PostgreSQL RAW (batches de 15k registros)
   ↓
4. Airflow DAG: data_processing
   ↓
5. PostgreSQL CLEAN (datos preprocesados)
   ↓
6. Airflow DAG: model_training
   ↓
7. MLflow (3 modelos: LR, RF, XGBoost)
   ↓
8. Airflow DAG: model_promotion
   ↓
9. MLflow Model Registry (stage: Production)
   ↓
10. FastAPI (carga modelo dinámicamente)
    ↓
11. Streamlit UI (usuario hace predicción)
    ↓
12. Prometheus (recolecta métricas)
    ↓
13. Grafana (visualiza métricas)
```

---

## 📊 COMPONENTES PRINCIPALES

### 1. Kubernetes (Infraestructura)
- **11 Deployments**: postgres-raw, postgres-clean, postgres-mlflow, minio, mlflow, airflow-webserver, airflow-scheduler, airflow-postgres, fastapi, streamlit, prometheus, grafana
- **11 Services**: Exponen cada componente
- **4 PersistentVolumeClaims**: Almacenamiento persistente

### 2. Airflow (Orquestación)
- **4 DAGs**:
  1. `data_ingestion_batch`: Cada 10 min, carga 15k registros
  2. `data_processing_pipeline`: Cada 15 min, transforma datos
  3. `model_training_pipeline`: Diario, entrena modelos
  4. `model_promotion_pipeline`: Diario, promociona mejor modelo

### 3. MLflow (Experimentos y Modelos)
- **Tracking**: Registro de métricas, parámetros, artefactos
- **Model Registry**: Versionado de modelos con stages (None, Staging, Production)
- **Artifacts**: Almacenados en MinIO (S3-compatible)
- **Metadata**: Almacenados en PostgreSQL

### 4. FastAPI (Inferencia)
- **Endpoints**:
  - `GET /`: Info del servicio
  - `GET /health`: Health check
  - `POST /predict`: Predicción de readmisión
  - `POST /reload_model`: Recarga modelo desde MLflow
  - `GET /metrics`: Métricas Prometheus
- **Features**:
  - Carga modelo dinámicamente desde stage "Production"
  - Métricas de Prometheus integradas
  - Sin hardcoding de modelos

### 5. Streamlit (UI)
- **Funcionalidades**:
  - Ingreso manual de datos del paciente
  - Valores predefinidos de ejemplo
  - Predicción con probabilidades por clase
  - Visualización de modelo en uso
  - Botón para recargar modelo

### 6. Prometheus + Grafana (Observabilidad)
- **Métricas**:
  - `predictions_total`: Contador por clase y versión
  - `prediction_latency_seconds`: Histograma de latencia
  - `prediction_errors_total`: Contador de errores
- **Dashboard Grafana**:
  - Requests totales
  - Latencia (p50, p95, p99)
  - Distribución de predicciones
  - Throughput

### 7. Locust (Load Testing)
- Simulación de usuarios concurrentes
- Generación de carga para métricas
- Tests configurables (users, spawn rate)

---

## 🔄 ESTRATEGIA DE DATOS

### Carga por Batches
- Dataset: ~100,000 registros
- Batch size: 15,000 registros
- Total batches: 7 cargas
- Frecuencia: Cada 10 minutos (automático)

### Split Estratificado
Cada batch se divide en:
- **Train**: 70% (10,500 registros)
- **Val**: 15% (2,250 registros)
- **Test**: 15% (2,250 registros)

**GARANTÍA**: Los datos de test de batch 1 NUNCA se usan en entrenamiento.

### Preprocesamiento
- **Remover**: encounter_id, patient_nbr, weight
- **Imputar**: race, payer_code, medical_specialty → "Unknown"
- **Transformar**:
  - age: intervalos → numérico + RobustScaler
  - diag_1/2/3: códigos ICD-9 → categorías clínicas
  - max_glu_serum, A1Cresult: ordinales (0,1,2,3)
  - medicamentos (23): ordinales (No=0, Steady=1, Down=2, Up=3)
  - change, diabetesMed: binarias (0, 1)
- **Encoding**:
  - OneHotEncoding: race, gender, admission IDs
  - Target: NO=0, >30=1, <30=2

### Modelos Entrenados
1. **LogisticRegression**: Baseline rápido
2. **RandomForest**: Balance precisión/velocidad
3. **XGBoost**: Máxima performance

**Métrica de selección**: F1-Score Weighted (balance entre clases)

---

## ✅ VALIDACIÓN (CHECKLIST)

### Despliegue Kubernetes (20%)
- [ ] `kubectl get pods -n mlops-proyecto3` → Todos "Running"
- [ ] Servicios accesibles via NodePort
- [ ] Screenshot de infraestructura completa

### MLflow con Bucket y PostgreSQL (20%)
- [ ] Experimento "diabetes_readmission" con 3 runs
- [ ] Artefactos en MinIO bucket "mlflow"
- [ ] Metadata en PostgreSQL postgres-mlflow
- [ ] Screenshot de MLflow UI

### Inferencia desde Production (20%)
- [ ] Modelo en stage "Production" en Model Registry
- [ ] API responde con `model_version` en JSON
- [ ] Cambio de modelo sin modificar código
- [ ] Screenshot de predicción

### Orquestación Airflow (20%)
- [ ] 4 DAGs activos y ejecutándose
- [ ] Datos en PostgreSQL RAW por batches
- [ ] Datos en PostgreSQL CLEAN procesados
- [ ] Screenshot de Airflow UI + logs

### Observabilidad (10%)
- [ ] Prometheus recolectando métricas de FastAPI
- [ ] Grafana con dashboard funcional
- [ ] Locust ejecutando tests con 100 usuarios
- [ ] Screenshot de métricas en Grafana

### Video (10%)
- [ ] Duración ≤ 10 minutos
- [ ] Explicar arquitectura (2 min)
- [ ] Demo procesamiento/experimentación (3 min)
- [ ] Demo UI (2 min)
- [ ] Explicar métricas (3 min)

---

## 🆘 PROBLEMAS COMUNES

### Pods no inician
```bash
kubectl logs <pod-name> -n mlops-proyecto3
kubectl describe pod <pod-name> -n mlops-proyecto3
```

### Servicios no accesibles
```bash
minikube service <service-name> -n mlops-proyecto3 --url
# O usar port-forward:
kubectl port-forward service/<service-name> <local-port>:<service-port> -n mlops-proyecto3
```

### MLflow no carga modelo
```bash
# Verificar modelo en Production
minikube service mlflow -n mlops-proyecto3
# UI → Model Registry → Verificar stage
```

### Airflow DAG no ejecuta
```bash
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3
# Verificar ConfigMap con variables de entorno
```

---

## 📚 DOCUMENTACIÓN ADICIONAL

- **README.md**: Instrucciones completas de instalación
- **docs/EVIDENCIAS.md**: Guía de validación y evidencias
- **docs/QUICK_REFERENCE.md**: Comandos rápidos
- **docs/WINDOWS_GITBASH.md**: Instrucciones específicas Windows

---

## 🎓 APRENDIZAJES CLAVE

1. **Kubernetes**: Orquestación de microservicios
2. **Airflow**: Pipelines de datos automatizados
3. **MLflow**: Gestión de experimentos y modelos
4. **Modelo dinámico**: Sin hardcoding en producción
5. **Observabilidad**: Prometheus + Grafana
6. **Load testing**: Locust para performance
7. **Separación de datos**: Garantía de no contaminación
8. **Estratificación**: Balance de clases en splits

---

## 📞 PRÓXIMOS PASOS

1. ✅ Desplegar sistema completo
2. ✅ Activar DAGs en Airflow
3. ✅ Esperar carga de datos (~70 min)
4. ✅ Entrenar modelos
5. ✅ Promocionar mejor modelo
6. ✅ Probar predicciones
7. ✅ Ejecutar tests de carga
8. ✅ Capturar evidencias
9. ✅ Grabar video de sustentación
10. ✅ Subir a YouTube y entregar

**¡ÉXITO EN TU PROYECTO!** 🚀
