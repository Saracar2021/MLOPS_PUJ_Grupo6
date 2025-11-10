# MLOps Proyecto 3 - Sistema de Predicción de Readmisión Hospitalaria
## Grupo 6 - Pontificia Universidad Javeriana

Sistema MLOps completo para predicción de readmisión hospitalaria en pacientes diabéticos utilizando Kubernetes, Airflow, MLflow, Prometheus y Grafana.

---

## 📋 REQUISITOS DEL SISTEMA

- **RAM**: 32GB (asignando 16GB a Minikube)
- **CPU**: 4+ cores
- **Disco**: 30GB libres
- **SO**: Windows con Git Bash
- **Software requerido**: Docker Desktop, Minikube, kubectl

---

## 🔧 INSTALACIÓN DE PREREQUISITOS

### 1. Instalar Docker Desktop (si no lo tienes)

```bash
# Descargar desde: https://www.docker.com/products/docker-desktop
# Instalar y reiniciar el sistema
# Verificar instalación:
docker --version
docker-compose --version
```

### 2. Instalar Minikube en Windows

```bash
# Descargar Minikube
curl -Lo minikube.exe https://github.com/kubernetes/minikube/releases/latest/download/minikube-windows-amd64.exe

# Mover a directorio accesible (ajusta la ruta según tu sistema)
mkdir -p ~/bin
mv minikube.exe ~/bin/minikube.exe

# Agregar al PATH (en Git Bash)
echo 'export PATH=$PATH:~/bin' >> ~/.bashrc
source ~/.bashrc

# Verificar instalación
minikube version
```

### 3. Instalar kubectl

```bash
# Descargar kubectl
curl -Lo kubectl.exe https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe

# Mover al mismo directorio
mv kubectl.exe ~/bin/kubectl.exe

# Verificar instalación
kubectl version --client
```

---

## 🚀 DESPLIEGUE DEL PROYECTO

### Paso 1: Iniciar Minikube

```bash
# Iniciar Minikube con recursos adecuados
minikube start --memory=16384 --cpus=6 --driver=docker

# Verificar que está corriendo
minikube status

# Habilitar addons necesarios
minikube addons enable ingress
minikube addons enable metrics-server
```

### Paso 2: Crear Namespace

```bash
kubectl create namespace mlops-proyecto3
kubectl config set-context --current --namespace=mlops-proyecto3
```

### Paso 3: Desplegar Bases de Datos

```bash
# PostgreSQL para RAW DATA
kubectl apply -f k8s/postgres-raw/

# PostgreSQL para CLEAN DATA
kubectl apply -f k8s/postgres-clean/

# PostgreSQL para MLflow metadata
kubectl apply -f k8s/postgres-mlflow/

# Esperar que estén ready (puede tomar 2-3 minutos)
kubectl wait --for=condition=ready pod -l app=postgres-raw --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres-clean --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres-mlflow --timeout=300s
```

### Paso 4: Desplegar MinIO (Object Storage)

```bash
kubectl apply -f k8s/minio/
kubectl wait --for=condition=ready pod -l app=minio --timeout=300s
```

### Paso 5: Desplegar MLflow

```bash
kubectl apply -f k8s/mlflow/
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s
```

### Paso 6: Desplegar Airflow

```bash
# Crear ConfigMaps y Secrets
kubectl apply -f k8s/airflow/configmap.yaml
kubectl apply -f k8s/airflow/secret.yaml

# Desplegar Airflow
kubectl apply -f k8s/airflow/deployment.yaml
kubectl apply -f k8s/airflow/service.yaml

# Esperar que esté ready
kubectl wait --for=condition=ready pod -l app=airflow-webserver --timeout=300s
```

### Paso 7: Desplegar FastAPI

```bash
kubectl apply -f k8s/fastapi/
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=300s
```

### Paso 8: Desplegar Streamlit UI

```bash
kubectl apply -f k8s/streamlit/
kubectl wait --for=condition=ready pod -l app=streamlit --timeout=300s
```

### Paso 9: Desplegar Observabilidad (Prometheus + Grafana)

```bash
kubectl apply -f k8s/prometheus/
kubectl apply -f k8s/grafana/
kubectl wait --for=condition=ready pod -l app=prometheus --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana --timeout=300s
```

### Paso 10: Exponer Servicios

```bash
# Exponer todos los servicios via NodePort
kubectl get services

# Obtener URLs de acceso
minikube service airflow-webserver -n mlops-proyecto3 --url
minikube service mlflow -n mlops-proyecto3 --url
minikube service fastapi -n mlops-proyecto3 --url
minikube service streamlit -n mlops-proyecto3 --url
minikube service grafana -n mlops-proyecto3 --url
```

---

## 🌐 ACCESO A LOS SERVICIOS

Una vez desplegado, accede a:

| Servicio | Comando | Credenciales |
|----------|---------|--------------|
| **Airflow** | `minikube service airflow-webserver --url` | admin / admin |
| **MLflow** | `minikube service mlflow --url` | Sin auth |
| **Streamlit** | `minikube service streamlit --url` | Sin auth |
| **Grafana** | `minikube service grafana --url` | admin / admin |
| **FastAPI Docs** | `minikube service fastapi --url` + `/docs` | Sin auth |

---

## 📊 EJECUCIÓN DEL PIPELINE

### 1. Carga de Datos por Batch

```bash
# Acceder a Airflow Web UI
# URL obtenida con: minikube service airflow-webserver --url

# Activar el DAG: data_ingestion_batch
# Este DAG se ejecuta cada 10 minutos y carga 15,000 registros

# Monitorear ejecución:
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3
```

**Evidencia esperada**:
- 7 ejecuciones del DAG (100k registros / 15k = 7 batches)
- En PostgreSQL RAW: Tabla con ~100k registros
- Columnas: `batch_id`, `split_type`, `data`

### 2. Procesamiento de Datos

```bash
# Activar el DAG: data_processing_pipeline
# Este DAG transforma RAW → CLEAN

# Verificar datos procesados:
kubectl exec -it deployment/postgres-clean -- psql -U mlops_user -d clean_db -c "SELECT split_type, COUNT(*) FROM clean_data GROUP BY split_type;"

# Resultado esperado:
# split_type | count
# -----------+-------
# train      | 70000
# val        | 15000
# test       | 15000
```

### 3. Entrenamiento de Modelos

```bash
# Activar el DAG: model_training_pipeline
# Este DAG entrena 3 modelos: LogisticRegression, RandomForest, XGBoost

# Verificar en MLflow:
# Acceder a MLflow UI
# Ver experimento "diabetes_readmission"
# Deben aparecer 3 runs con métricas
```

**Evidencia esperada en MLflow**:
- Experimento: `diabetes_readmission`
- 3 runs registrados
- Métricas: `f1_score_weighted`, `accuracy`, `recall_class_2`
- Artefactos: modelo.pkl, confusion_matrix.png, feature_importance.png

### 4. Promoción a Producción

```bash
# Activar el DAG: model_promotion_pipeline
# Este DAG selecciona el mejor modelo (mayor F1-score) y lo marca como "Production"

# Verificar en MLflow:
# Model Registry → Modelos registrados
# Debe haber 1 modelo en stage "Production"
```

### 5. Inferencia via API

```bash
# Obtener URL de FastAPI
FASTAPI_URL=$(minikube service fastapi --url)

# Realizar predicción
curl -X POST "$FASTAPI_URL/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "race": "Caucasian",
    "gender": "Female",
    "age": "[50-60)",
    "time_in_hospital": 3,
    "num_lab_procedures": 45,
    "num_procedures": 2,
    "num_medications": 15,
    "number_diagnoses": 9,
    "max_glu_serum": "None",
    "A1Cresult": ">8",
    "diabetesMed": "yes",
    "change": "yes"
  }'

# Respuesta esperada:
# {
#   "prediction": "<30",
#   "probabilities": {
#     "NO": 0.15,
#     ">30": 0.25,
#     "<30": 0.60
#   },
#   "model_version": "RandomForest_v1.2.3",
#   "timestamp": "2025-11-09T..."
# }
```

### 6. Interfaz Gráfica (Streamlit)

```bash
# Acceder a Streamlit UI
# URL obtenida con: minikube service streamlit --url

# Funcionalidades:
# - Ingresar valores manualmente
# - Usar valores predefinidos
# - Ver predicción y probabilidades
# - Visualizar versión del modelo en uso
```

### 7. Pruebas de Carga con Locust

```bash
# Port-forward Locust
kubectl port-forward service/locust 8089:8089

# Acceder a: http://localhost:8089
# Configurar:
# - Number of users: 100
# - Spawn rate: 10
# - Host: http://fastapi:8000

# Iniciar test y observar métricas en tiempo real
```

### 8. Observabilidad con Prometheus + Grafana

```bash
# Acceder a Grafana
GRAFANA_URL=$(minikube service grafana --url)

# Login: admin / admin
# Importar dashboard: k8s/grafana/dashboard.json
```

**Métricas esperadas**:
- Requests totales a `/predict`
- Latencia (p50, p95, p99)
- Errores 4xx/5xx
- Predicciones por clase
- Uso de CPU/memoria de FastAPI pod

---

## ✅ CHECKLIST DE EVIDENCIAS (20% c/u)

### 1. Despliegue Kubernetes (20%)
- [ ] Todos los pods en estado `Running`
- [ ] Servicios expuestos correctamente
- [ ] Screenshot de `kubectl get all -n mlops-proyecto3`

### 2. MLflow con Bucket y PostgreSQL (20%)
- [ ] Experimentos registrados
- [ ] Modelos en Model Registry
- [ ] Artefactos en MinIO
- [ ] Screenshot de MLflow UI mostrando runs

### 3. Inferencia desde Production (20%)
- [ ] API consume modelo de MLflow stage "Production"
- [ ] Cambio de modelo sin modificar código
- [ ] Screenshot de respuesta JSON con `model_version`

### 4. Orquestación con Airflow (20%)
- [ ] 4 DAGs funcionando
- [ ] Logs de ejecución exitosa
- [ ] Screenshot de Airflow UI con DAG runs

### 5. Observabilidad (10%)
- [ ] Prometheus recolectando métricas
- [ ] Dashboard Grafana funcional
- [ ] Screenshot de métricas durante test Locust

### 6. Video Sustentación (10%)
- [ ] Explicar arquitectura (2 min)
- [ ] Mostrar procesamiento y experimentación (3 min)
- [ ] Demo de UI (2 min)
- [ ] Explicar métricas y dashboards (3 min)
- [ ] Subir a YouTube (máx 10 min)

---

## 🛠️ COMANDOS ÚTILES

### Ver logs
```bash
# Logs de un pod específico
kubectl logs -f deployment/mlflow

# Logs de Airflow scheduler
kubectl logs -f deployment/airflow-scheduler

# Logs de FastAPI
kubectl logs -f deployment/fastapi
```

### Verificar datos
```bash
# Conectar a PostgreSQL RAW
kubectl exec -it deployment/postgres-raw -- psql -U mlops_user -d raw_db

# Query de ejemplo
SELECT batch_id, split_type, COUNT(*) FROM raw_data GROUP BY batch_id, split_type;
```

### Reiniciar servicios
```bash
# Reiniciar deployment
kubectl rollout restart deployment/fastapi

# Escalar pods
kubectl scale deployment/fastapi --replicas=3
```

### Detener todo
```bash
# Eliminar namespace completo
kubectl delete namespace mlops-proyecto3

# Detener Minikube
minikube stop

# Eliminar Minikube (limpieza total)
minikube delete
```

---

## 📁 ESTRUCTURA DEL PROYECTO

```
proyecto3_mlops/
├── k8s/                          # Manifiestos Kubernetes
│   ├── namespace.yaml
│   ├── postgres-raw/
│   ├── postgres-clean/
│   ├── postgres-mlflow/
│   ├── minio/
│   ├── mlflow/
│   ├── airflow/
│   ├── fastapi/
│   ├── streamlit/
│   ├── prometheus/
│   └── grafana/
├── airflow/
│   └── dags/
│       ├── data_ingestion_batch.py
│       ├── data_processing.py
│       ├── model_training.py
│       └── model_promotion.py
├── ml/
│   ├── preprocessing.py
│   ├── train.py
│   └── evaluate.py
├── api/
│   ├── main.py
│   └── model_loader.py
├── ui/
│   └── app.py
├── locust/
│   └── locustfile.py
├── docs/
│   └── EVIDENCIAS.md
└── README.md
```

---

## 🆘 TROUBLESHOOTING

### Error: Minikube no inicia
```bash
# Verificar Docker Desktop está corriendo
docker ps

# Reiniciar Minikube
minikube delete
minikube start --memory=16384 --cpus=6 --driver=docker
```

### Error: Pod en CrashLoopBackOff
```bash
# Ver logs del pod
kubectl logs <pod-name>

# Describir pod para ver eventos
kubectl describe pod <pod-name>
```

### Error: Servicios no accesibles
```bash
# Verificar que minikube tunnel está corriendo
minikube tunnel

# Alternativamente usar port-forward
kubectl port-forward service/mlflow 5000:5000
```

---

## 📞 CONTACTO

Grupo 6 - Pontificia Universidad Javeriana
Curso: MLOps 2025-2
