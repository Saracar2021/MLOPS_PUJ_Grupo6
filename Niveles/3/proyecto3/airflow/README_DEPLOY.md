# Despliegue de Airflow Corregido

## Problemas Corregidos

### 1. Falta de Dependencias
**Problema:** La imagen base de Airflow no incluía las librerías necesarias para ejecutar los DAGs de ML.

**Solución:** Se creó un `Dockerfile` personalizado que extiende la imagen base e instala todas las dependencias:
- mlflow==2.8.0
- scikit-learn==1.3.2
- xgboost==2.0.2
- pandas==2.1.3
- psycopg2-binary==2.9.9
- matplotlib==3.8.2
- seaborn==0.13.0

### 2. Error en model_training.py
**Problema:** El DAG usaba `mlflow.sklearn.log_model()` para registrar modelos XGBoost, lo cual es incorrecto.

**Solución:** Se corrigió para usar `mlflow.xgboost.log_model()` específicamente para modelos XGBoost.

---

## Instrucciones de Despliegue

### Paso 1: Construir la Imagen Personalizada

```bash
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

Este script construirá la imagen Docker personalizada llamada `custom-airflow:2.7.0-mlops`.

**Nota:** Si usas Minikube o Docker Desktop con Kubernetes, asegúrate de que Docker esté configurado para usar el mismo contexto:

Para Minikube:
```bash
eval $(minikube docker-env)
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

Para Docker Desktop:
```bash
# La imagen se construirá en tu Docker local
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

### Paso 2: Verificar la Imagen

```bash
docker images | grep custom-airflow
```

Deberías ver:
```
custom-airflow   2.7.0-mlops   <image-id>   <time>   <size>
```

### Paso 3: Redesplegar Airflow en Kubernetes

Si Airflow ya está desplegado, primero elimínalo:

```bash
kubectl delete -f ../k8s/airflow/all-in-one.yaml
```

Espera unos segundos y luego redespliega con la nueva configuración:

```bash
kubectl apply -f ../k8s/airflow/all-in-one.yaml
```

### Paso 4: Verificar el Despliegue

Verifica que los pods de Airflow estén corriendo:

```bash
kubectl get pods -n mlops-proyecto3 | grep airflow
```

Verifica los logs del scheduler para confirmar que las dependencias están instaladas:

```bash
kubectl logs -n mlops-proyecto3 deployment/airflow-scheduler | head -20
```

### Paso 5: Acceder a Airflow UI

```bash
# Si usas NodePort (configuración actual)
# Accede a: http://<node-ip>:30800

# Para obtener la URL en Minikube:
minikube service airflow-webserver -n mlops-proyecto3 --url
```

Credenciales por defecto:
- **Usuario:** admin
- **Contraseña:** admin

### Paso 6: Ejecutar el DAG

1. Accede a la UI de Airflow
2. Busca el DAG `model_training_pipeline`
3. Actívalo (toggle ON)
4. Haz clic en "Trigger DAG" para ejecutarlo manualmente

---

## Troubleshooting

### Error: ImagePullBackOff
Si ves este error, significa que Kubernetes no puede encontrar la imagen. Asegúrate de:
- Haber construido la imagen en el mismo contexto de Docker que usa Kubernetes
- Usar `imagePullPolicy: IfNotPresent` (ya configurado)

Para Minikube:
```bash
eval $(minikube docker-env)
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

### Error: CrashLoopBackOff
Revisa los logs del pod para ver el error:
```bash
kubectl logs -n mlops-proyecto3 <pod-name>
```

### Error: Base de datos no conecta
Verifica que los servicios de PostgreSQL estén corriendo:
```bash
kubectl get pods -n mlops-proyecto3 | grep postgres
```

---

## Estructura de Archivos

```
airflow/
├── Dockerfile              # Imagen personalizada de Airflow
├── requirements.txt        # Dependencias Python
├── build-image.sh         # Script para construir la imagen
├── README_DEPLOY.md       # Este archivo
└── dags/
    ├── model_training.py  # DAG corregido
    ├── data_processing.py
    ├── data_ingestion_batch.py
    └── model_promotion.py
```

---

## Uso en Producción

Para un entorno de producción, considera:

1. **Registry Privado:**
   ```bash
   # Tagear la imagen para tu registry
   docker tag custom-airflow:2.7.0-mlops your-registry.com/custom-airflow:2.7.0-mlops

   # Push al registry
   docker push your-registry.com/custom-airflow:2.7.0-mlops
   ```

2. **Actualizar el all-in-one.yaml:**
   ```yaml
   image: your-registry.com/custom-airflow:2.7.0-mlops
   imagePullPolicy: Always
   ```

3. **Secrets para el Registry:**
   ```bash
   kubectl create secret docker-registry regcred \
     --docker-server=your-registry.com \
     --docker-username=<username> \
     --docker-password=<password> \
     -n mlops-proyecto3
   ```

   Y añadir al deployment:
   ```yaml
   spec:
     imagePullSecrets:
     - name: regcred
   ```

---

## Próximos Pasos

1. ✅ Construir la imagen personalizada
2. ✅ Redesplegar Airflow
3. ✅ Ejecutar el DAG `model_training_pipeline`
4. Verificar que los modelos se registren correctamente en MLflow
5. Validar las métricas en MLflow UI

Para más información sobre el proyecto, consulta el `README.md` principal.
