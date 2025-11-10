# QUICK REFERENCE - Comandos Rápidos

## Iniciar/Detener Minikube

```bash
# Iniciar
minikube start --memory=16384 --cpus=6 --driver=docker

# Detener
minikube stop

# Eliminar (reset completo)
minikube delete
```

## Despliegue Completo

```bash
# Opción 1: Script automatizado
./deploy.sh

# Opción 2: Manual
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

## Acceso a Servicios

```bash
# Ver todas las URLs
minikube service list -n mlops-proyecto3

# Airflow
minikube service airflow-webserver -n mlops-proyecto3 --url

# MLflow
minikube service mlflow -n mlops-proyecto3 --url

# Streamlit
minikube service streamlit -n mlops-proyecto3 --url

# FastAPI
minikube service fastapi -n mlops-proyecto3 --url

# Grafana
minikube service grafana -n mlops-proyecto3 --url
```

## Verificación de Estado

```bash
# Ver todos los recursos
kubectl get all -n mlops-proyecto3

# Ver pods con detalles
kubectl get pods -n mlops-proyecto3 -o wide

# Ver logs de un pod
kubectl logs -f deployment/mlflow -n mlops-proyecto3

# Ver eventos
kubectl get events -n mlops-proyecto3 --sort-by='.lastTimestamp'

# Describir pod (para debugging)
kubectl describe pod <pod-name> -n mlops-proyecto3
```

## Validación de Datos

```bash
# PostgreSQL RAW
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- psql -U mlops_user -d raw_db

# Queries útiles:
SELECT batch_id, split_type, COUNT(*) FROM raw_data GROUP BY batch_id, split_type;
SELECT COUNT(*) FROM raw_data;
SELECT DISTINCT batch_id FROM raw_data ORDER BY batch_id;

# PostgreSQL CLEAN
kubectl exec -it deployment/postgres-clean -n mlops-proyecto3 -- psql -U mlops_user -d clean_db

# Queries útiles:
SELECT split_type, COUNT(*) FROM clean_data GROUP BY split_type;
SELECT batch_id, COUNT(*) FROM clean_data GROUP BY batch_id;

# PostgreSQL MLflow
kubectl exec -it deployment/postgres-mlflow -n mlops-proyecto3 -- psql -U mlflow_user -d mlflow_db

# Queries útiles:
\dt
SELECT COUNT(*) FROM experiments;
SELECT COUNT(*) FROM runs;
```

## Test de API

```bash
# Health check
FASTAPI_URL=$(minikube service fastapi -n mlops-proyecto3 --url)
curl $FASTAPI_URL/health

# Predicción
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
    "change": "yes",
    "admission_type_id": 1,
    "discharge_disposition_id": 1,
    "admission_source_id": 7,
    "payer_code": "MC",
    "medical_specialty": "Cardiology",
    "diag_1": "250.83",
    "diag_2": "401.9",
    "diag_3": "428.0",
    "number_outpatient": 0,
    "number_emergency": 0,
    "number_inpatient": 1
  }'

# Métricas Prometheus
curl $FASTAPI_URL/metrics

# Reload model
curl -X POST "$FASTAPI_URL/reload_model"
```

## Airflow DAGs

```bash
# Ver logs del scheduler
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3

# Trigger manual de DAG (desde UI)
# 1. Acceder a Airflow UI
# 2. Activar DAG con el toggle
# 3. Click en "Trigger DAG" (play button)
```

## Locust Load Testing

```bash
# Port-forward Locust
kubectl port-forward service/locust 8089:8089 -n mlops-proyecto3

# Acceder a: http://localhost:8089
# Number of users: 100
# Spawn rate: 10
# Host: http://fastapi:8000
```

## Prometheus y Grafana

```bash
# Prometheus UI
minikube service prometheus -n mlops-proyecto3 --url

# Grafana UI
minikube service grafana -n mlops-proyecto3 --url
# Login: admin/admin

# Queries útiles en Prometheus:
rate(predictions_total[5m])
histogram_quantile(0.95, rate(prediction_latency_seconds_bucket[5m]))
prediction_errors_total
```

## Limpieza

```bash
# Eliminar deployment específico
kubectl delete deployment <name> -n mlops-proyecto3

# Eliminar todo el namespace
kubectl delete namespace mlops-proyecto3

# Reset completo de Minikube
minikube delete
rm -rf ~/.minikube
```

## Troubleshooting

```bash
# Pod en CrashLoopBackOff
kubectl logs <pod-name> -n mlops-proyecto3
kubectl describe pod <pod-name> -n mlops-proyecto3

# Ver recursos del nodo
kubectl top nodes
kubectl top pods -n mlops-proyecto3

# Reiniciar deployment
kubectl rollout restart deployment/<name> -n mlops-proyecto3

# Escalar replicas
kubectl scale deployment/<name> --replicas=2 -n mlops-proyecto3

# Port-forward para debugging
kubectl port-forward service/<service-name> <local-port>:<service-port> -n mlops-proyecto3
```

## Exportar Evidencias

```bash
# Exportar estado completo
kubectl get all -n mlops-proyecto3 > evidencia_k8s.txt

# Exportar pods
kubectl get pods -n mlops-proyecto3 -o wide > evidencia_pods.txt

# Exportar datos RAW
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- psql -U mlops_user -d raw_db -c "SELECT * FROM batch_control;" > evidencia_batches.txt

# Exportar métricas
FASTAPI_URL=$(minikube service fastapi -n mlops-proyecto3 --url)
curl $FASTAPI_URL/metrics > evidencia_metrics.txt

# Screenshot de MLflow experiments
MLFLOW_URL=$(minikube service mlflow -n mlops-proyecto3 --url)
curl "$MLFLOW_URL/api/2.0/mlflow/experiments/search" > evidencia_experiments.json
```

## Variables de Entorno Importantes

```bash
# En Airflow
MLFLOW_TRACKING_URI=http://mlflow:5000
POSTGRES_RAW_HOST=postgres-raw
POSTGRES_CLEAN_HOST=postgres-clean

# En FastAPI
MLFLOW_TRACKING_URI=http://mlflow:5000
AWS_ACCESS_KEY_ID=minioadmin
AWS_SECRET_ACCESS_KEY=minioadmin
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# En Streamlit
FASTAPI_URL=http://fastapi:8000
MLFLOW_URL=http://mlflow:5000
```

## Checklist de Validación

- [ ] Minikube running
- [ ] Todos los pods en "Running"
- [ ] Servicios accesibles (airflow, mlflow, streamlit)
- [ ] Al menos 3 batches cargados en RAW
- [ ] Datos procesados en CLEAN
- [ ] 3 modelos entrenados en MLflow
- [ ] 1 modelo en "Production"
- [ ] API responde a /predict
- [ ] Streamlit accesible y funcional
- [ ] Prometheus recolectando métricas
- [ ] Grafana con dashboard
- [ ] Locust ejecutando tests
