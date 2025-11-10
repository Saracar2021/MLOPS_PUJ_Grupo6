# EVIDENCIAS Y VALIDACIÓN DEL PROYECTO 3

## DISTRIBUCIÓN DE PUNTAJE Y EVIDENCIAS REQUERIDAS

### 1. Despliegue Kubernetes (20%)

**Requisito**: Sistema completo desplegado en K8s exponiendo interfaz gráfica

**Comandos de validación**:
```bash
# Ver todos los pods running
kubectl get pods -n mlops-proyecto3

# Resultado esperado: Todos los pods en estado "Running"
NAME                                READY   STATUS    RESTARTS   AGE
postgres-raw-xxx                    1/1     Running   0          5m
postgres-clean-xxx                  1/1     Running   0          5m
postgres-mlflow-xxx                 1/1     Running   0          5m
minio-xxx                           1/1     Running   0          4m
mlflow-xxx                          1/1     Running   0          3m
airflow-webserver-xxx               1/1     Running   0          3m
airflow-scheduler-xxx               1/1     Running   0          3m
fastapi-xxx                         1/1     Running   0          2m
streamlit-xxx                       1/1     Running   0          2m
prometheus-xxx                      1/1     Running   0          2m
grafana-xxx                         1/1     Running   0          2m

# Ver todos los servicios
kubectl get services -n mlops-proyecto3

# Acceder a servicios
minikube service streamlit -n mlops-proyecto3 --url
```

**Evidencia para video**:
- Screenshot de `kubectl get all -n mlops-proyecto3`
- Screenshot de Streamlit UI accesible desde navegador

---

### 2. MLflow con Bucket y PostgreSQL (20%)

**Requisito**: Tracking de experimentos con bucket (MinIO) + PostgreSQL para metadatos

**Comandos de validación**:
```bash
# Acceder a MLflow UI
minikube service mlflow -n mlops-proyecto3 --url

# Verificar conexión a PostgreSQL metadata
kubectl exec -it deployment/postgres-mlflow -n mlops-proyecto3 -- psql -U mlflow_user -d mlflow_db -c "\dt"

# Resultado esperado: Tablas de MLflow (experiments, runs, metrics, params, tags, artifacts)

# Verificar bucket MinIO
minikube service minio -n mlops-proyecto3 --url
# Login con minioadmin/minioadmin
# Verificar bucket "mlflow" con artefactos
```

**Evidencia para video**:
- Screenshot de MLflow UI mostrando experimento "diabetes_readmission"
- Screenshot de 3 runs con métricas (f1_score_weighted, accuracy, recall_class_2)
- Screenshot de artefactos en MinIO (modelos, confusion_matrix.png)
- Screenshot de PostgreSQL con tablas de metadata

**Validación de experimentos**:
```bash
# Conectar a PostgreSQL MLflow
kubectl exec -it deployment/postgres-mlflow -n mlops-proyecto3 -- psql -U mlflow_user -d mlflow_db

# Query de validación
SELECT r.run_id, r.run_name, m.key, m.value 
FROM runs r 
JOIN metrics m ON r.run_id = m.run_id 
WHERE m.key = 'f1_score_weighted'
ORDER BY CAST(m.value AS FLOAT) DESC;

# Resultado esperado: 3 runs con sus scores
```

---

### 3. Inferencia desde Production (20%)

**Requisito**: API consume modelo de MLflow stage "Production" SIN cambios en código

**Comandos de validación**:
```bash
# Verificar modelo en Production
minikube service mlflow -n mlops-proyecto3 --url
# Navegar a Model Registry
# Verificar que existe modelo "diabetes_readmission_model" en stage "Production"

# Test de predicción
FASTAPI_URL=$(minikube service fastapi -n mlops-proyecto3 --url)
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

# Resultado esperado:
{
  "prediction": "<30" | ">30" | "NO",
  "probabilities": {
    "NO": 0.xx,
    ">30": 0.xx,
    "<30": 0.xx
  },
  "model_version": "diabetes_readmission_model_vX",
  "timestamp": "2025-11-09T..."
}

# Verificar cambio dinámico de modelo
# 1. Entrenar nuevo modelo con mejores parámetros
# 2. Promover a Production
# 3. Llamar endpoint /reload_model
curl -X POST "$FASTAPI_URL/reload_model"
# 4. Hacer nueva predicción - debe usar nuevo modelo
```

**Evidencia para video**:
- Screenshot de Model Registry con modelo en "Production"
- Screenshot de respuesta JSON mostrando `model_version`
- Demo de cambio de modelo (antes/después de promoción)

---

### 4. Orquestación con Airflow (20%)

**Requisito**: DAGs para recolección, procesamiento, almacenamiento y entrenamiento

**DAGs implementados**:
1. `data_ingestion_batch` - Carga datos en batches de 15k (cada 10 min)
2. `data_processing_pipeline` - Transforma RAW → CLEAN (cada 15 min)
3. `model_training_pipeline` - Entrena 3 modelos (diario)
4. `model_promotion_pipeline` - Promociona mejor modelo a Production (diario)

**Comandos de validación**:
```bash
# Acceder a Airflow UI
minikube service airflow-webserver -n mlops-proyecto3 --url
# Login: admin/admin

# Ver logs de ejecución
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3

# Verificar datos en RAW
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- psql -U mlops_user -d raw_db

SELECT batch_id, split_type, COUNT(*) 
FROM raw_data 
GROUP BY batch_id, split_type 
ORDER BY batch_id;

# Resultado esperado:
 batch_id | split_type | count
----------+------------+-------
        1 | train      | 10500
        1 | val        |  2250
        1 | test       |  2250
        2 | train      | 10500
        2 | val        |  2250
        2 | test       |  2250
        ...

# Verificar datos en CLEAN
kubectl exec -it deployment/postgres-clean -n mlops-proyecto3 -- psql -U mlops_user -d clean_db

SELECT split_type, COUNT(*) 
FROM clean_data 
GROUP BY split_type;

# Resultado esperado:
 split_type | count
------------+-------
 train      | 70000
 val        | 15000
 test       | 15000
```

**Evidencia para video**:
- Screenshot de Airflow UI mostrando 4 DAGs activos
- Screenshot de Graph View de cada DAG
- Screenshot de logs exitosos de ejecución
- Screenshot de queries en PostgreSQL mostrando datos por batch

---

### 5. Observabilidad (10%)

**Requisito**: Prometheus + Grafana sobre API, pruebas con Locust

**Comandos de validación**:
```bash
# Acceder a Prometheus
minikube service prometheus -n mlops-proyecto3 --url
# Verificar targets: fastapi:8000 debe estar UP

# Acceder a Grafana
minikube service grafana -n mlops-proyecto3 --url
# Login: admin/admin

# Ejecutar Locust
kubectl port-forward service/locust 8089:8089 -n mlops-proyecto3
# Acceder a http://localhost:8089
# Configurar: 100 users, spawn rate 10, host: http://fastapi:8000
# Iniciar test

# Ver métricas en tiempo real
curl $(minikube service fastapi -n mlops-proyecto3 --url)/metrics

# Resultado esperado: Métricas de Prometheus
# predictions_total{model_version="...",prediction_class="<30"} 45
# predictions_total{model_version="...",prediction_class=">30"} 30
# predictions_total{model_version="...",prediction_class="NO"} 25
# prediction_latency_seconds_sum 12.5
# prediction_latency_seconds_count 100
# prediction_errors_total 0
```

**Métricas esperadas en Grafana**:
- Requests totales
- Latencia (p50, p95, p99)
- Errores 4xx/5xx
- Predicciones por clase
- Throughput (requests/sec)

**Evidencia para video**:
- Screenshot de Prometheus targets
- Screenshot de Grafana dashboard con métricas
- Screenshot de Locust ejecutando test con 100 usuarios
- Screenshot de métricas durante carga

---

### 6. Video Sustentación (10%)

**Estructura recomendada (10 minutos)**:

**Minutos 0-2: Arquitectura**
- Diagrama de componentes en Kubernetes
- Explicar flujo: Ingesta → Procesamiento → Entrenamiento → Producción → Inferencia
- Mostrar `kubectl get all` con todos los pods corriendo

**Minutos 2-5: Procesamiento y Experimentación**
- Demostrar Airflow UI con DAGs
- Explicar estrategia de batches (15k registros, stratified split)
- Mostrar MLflow con 3 experimentos registrados
- Explicar transformaciones de preprocesamiento

**Minutos 5-7: Demostración de UI**
- Acceder a Streamlit
- Ingresar datos de paciente
- Realizar predicción
- Mostrar resultado con probabilidades
- Mostrar versión del modelo en uso

**Minutos 7-10: Métricas y Dashboards**
- Acceder a Grafana
- Explicar métricas recolectadas
- Mostrar dashboard durante test Locust
- Explicar latencias y throughput
- Conclusiones y aprendizajes

---

## CHECKLIST FINAL DE VALIDACIÓN

### Antes del video:

- [ ] Minikube corriendo con todos los pods en "Running"
- [ ] Al menos 3 batches de datos cargados (45k registros)
- [ ] Datos procesados en CLEAN database
- [ ] 3 modelos entrenados en MLflow
- [ ] 1 modelo en stage "Production"
- [ ] Streamlit accesible desde navegador
- [ ] Prometheus recolectando métricas
- [ ] Grafana con dashboard configurado

### Durante el video:

- [ ] Mostrar arquitectura completa
- [ ] Ejecutar `kubectl get all`
- [ ] Demostrar Airflow con DAGs
- [ ] Mostrar MLflow con experimentos
- [ ] Hacer predicción en Streamlit
- [ ] Ejecutar test Locust
- [ ] Mostrar métricas en Grafana
- [ ] Explicar aprendizajes

### Comandos útiles para captura de evidencias:

```bash
# Screenshot de toda la infraestructura
kubectl get all -n mlops-proyecto3 > evidencia_k8s.txt

# Screenshot de datos
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- psql -U mlops_user -d raw_db -c "SELECT batch_id, COUNT(*) FROM raw_data GROUP BY batch_id;" > evidencia_raw.txt

# Screenshot de modelos
curl $(minikube service mlflow -n mlops-proyecto3 --url)/api/2.0/mlflow/registered-models/list > evidencia_models.json

# Screenshot de métricas
curl $(minikube service fastapi -n mlops-proyecto3 --url)/metrics > evidencia_metrics.txt
```

---

## TIPS PARA MAXIMIZAR PUNTAJE

1. **Despliegue (20%)**:
   - Todos los pods deben estar en "Running"
   - Servicios accesibles via NodePort
   - Screenshots claros de `kubectl get all`

2. **MLflow (20%)**:
   - Mostrar claramente experimentos con métricas
   - Evidenciar artifacts en MinIO
   - Query a PostgreSQL con metadata

3. **Producción (20%)**:
   - Demostrar cambio dinámico de modelo
   - JSON response con model_version
   - No hay hardcoding de modelos en código

4. **Airflow (20%)**:
   - 4 DAGs funcionando
   - Logs de ejecución exitosa
   - Datos verificables en databases

5. **Observabilidad (10%)**:
   - Métricas reales durante Locust
   - Dashboard funcional en Grafana
   - Capturas durante pruebas de carga

6. **Video (10%)**:
   - Calidad de audio/video
   - Explicación clara de arquitectura
   - Demostraciones funcionales
   - Tiempo ≤ 10 minutos

---

## TROUBLESHOOTING COMÚN

### Problema: Pods en CrashLoopBackOff
```bash
kubectl logs <pod-name> -n mlops-proyecto3
kubectl describe pod <pod-name> -n mlops-proyecto3
```

### Problema: MLflow no carga modelo
```bash
kubectl exec -it deployment/mlflow -n mlops-proyecto3 -- sh
# Verificar conectividad a PostgreSQL y MinIO
```

### Problema: Airflow DAG no se ejecuta
```bash
kubectl logs deployment/airflow-scheduler -n mlops-proyecto3 -f
# Verificar variables de entorno en ConfigMap
```

### Problema: FastAPI retorna 503
```bash
# Verificar que existe modelo en Production
kubectl exec -it deployment/mlflow -n mlops-proyecto3 -- sh
# curl http://localhost:5000/api/2.0/mlflow/registered-models/list
```
