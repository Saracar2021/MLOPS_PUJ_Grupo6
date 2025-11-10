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

### 3. DAGs No Visibles en Airflow UI
**Problema:** Los archivos DAG no se copiaban al PersistentVolumeClaim (PVC) usado por Airflow en Kubernetes, resultando en que no aparecía ningún DAG disponible.

**Solución:**
- Modificado el `Dockerfile` para incluir los DAGs en la imagen Docker en `/opt/airflow/dags-source/`
- Agregado un `initContainer` que copia los DAGs desde la imagen al PVC en cada despliegue
- Esto garantiza que los DAGs estén siempre disponibles para el webserver y scheduler

---

## Instrucciones de Despliegue

### Opción 1: Script Automatizado (Recomendado)

Desde el directorio raíz del proyecto3:

```bash
cd Niveles/3/proyecto3
./redeploy-airflow.sh
```

Este script automáticamente:
1. Configura Docker para usar el daemon de Minikube
2. Construye la imagen personalizada de Airflow con los DAGs incluidos
3. Elimina el despliegue anterior de Airflow
4. Despliega la nueva versión con los DAGs
5. Muestra la URL de acceso y comandos útiles de verificación

### Opción 2: Paso a Paso Manual

#### Paso 1: Configurar Docker para Minikube

**IMPORTANTE:** Si usas Minikube, DEBES configurar Docker para usar el daemon de Minikube:

```bash
eval $(minikube docker-env)
```

Este comando debe ejecutarse en cada nueva terminal que uses. Sin esto, la imagen se construirá en tu Docker local y Kubernetes no podrá encontrarla.

#### Paso 2: Construir la Imagen Personalizada

```bash
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

Este script construirá la imagen Docker personalizada llamada `custom-airflow:2.7.0-mlops` que incluye:
- Todas las dependencias de Python necesarias
- Los archivos DAG en `/opt/airflow/dags-source/`

#### Paso 3: Verificar la Imagen

```bash
docker images | grep custom-airflow
```

Deberías ver:
```
custom-airflow   2.7.0-mlops   <image-id>   <time>   <size>
```

#### Paso 4: Redesplegar Airflow en Kubernetes

Vuelve al directorio del proyecto:

```bash
cd ..
```

Si Airflow ya está desplegado, primero elimínalo:

```bash
kubectl delete -f k8s/airflow/all-in-one.yaml
```

Espera unos segundos y luego redespliega con la nueva configuración:

```bash
kubectl apply -f k8s/airflow/all-in-one.yaml
```

#### Paso 5: Verificar el Despliegue

Verifica que los pods de Airflow estén corriendo:

```bash
kubectl get pods -n mlops-proyecto3 | grep airflow
```

**IMPORTANTE:** Verifica que los DAGs se copiaron correctamente al PVC:

```bash
kubectl exec -it deployment/airflow-webserver -n mlops-proyecto3 -- ls -la /opt/airflow/dags/
```

Deberías ver los siguientes archivos:
- `data_ingestion_batch.py`
- `data_processing.py`
- `model_training.py`
- `model_promotion.py`

Verifica los logs del scheduler para confirmar que los DAGs se cargaron:

```bash
kubectl logs -n mlops-proyecto3 deployment/airflow-scheduler
```

Busca líneas como:
```
[2024-XX-XX] {dagbag.py:XXX} INFO - Filling up the DagBag from /opt/airflow/dags
```

#### Paso 6: Acceder a Airflow UI

```bash
# Si usas NodePort (configuración actual)
# Accede a: http://<node-ip>:30800

# Para obtener la URL en Minikube:
minikube service airflow-webserver -n mlops-proyecto3 --url
```

Credenciales por defecto:
- **Usuario:** admin
- **Contraseña:** admin

#### Paso 7: Ejecutar el DAG

1. Accede a la UI de Airflow
2. Deberías ver los siguientes DAGs:
   - `data_ingestion_batch`
   - `data_processing_pipeline`
   - `model_training_pipeline`
   - `model_promotion_pipeline`
3. Activa los DAGs que quieras ejecutar (toggle ON)
4. Haz clic en "Trigger DAG" para ejecutarlos manualmente

---

## Troubleshooting

### Problema: No aparecen DAGs en la UI

**Síntomas:**
- La UI de Airflow muestra "No DAGs found"
- El directorio `/opt/airflow/dags/` está vacío

**Soluciones:**

1. **Verificar que el initContainer copió los DAGs:**
   ```bash
   kubectl logs -n mlops-proyecto3 deployment/airflow-webserver -c copy-dags
   ```
   Deberías ver:
   ```
   Copying DAGs to PVC...
   DAGs copied successfully
   ```

2. **Verificar contenido del PVC:**
   ```bash
   kubectl exec -it deployment/airflow-webserver -n mlops-proyecto3 -- ls -la /opt/airflow/dags/
   ```
   Si no ves archivos `.py`, el problema es que los DAGs no se copiaron.

3. **Reconstruir y redesplegar:**
   ```bash
   eval $(minikube docker-env)
   cd Niveles/3/proyecto3
   ./redeploy-airflow.sh
   ```

4. **Verificar que la imagen contiene los DAGs:**
   ```bash
   docker run --rm custom-airflow:2.7.0-mlops ls -la /opt/airflow/dags-source/
   ```
   Deberías ver los 4 archivos de DAGs.

### Error: ImagePullBackOff

Si ves este error, significa que Kubernetes no puede encontrar la imagen.

**Causa:** La imagen se construyó en tu Docker local, no en el daemon de Minikube.

**Solución:**
```bash
eval $(minikube docker-env)
cd Niveles/3/proyecto3/airflow
./build-image.sh
```

Verifica que la imagen esté en Minikube:
```bash
eval $(minikube docker-env)
docker images | grep custom-airflow
```

### Error: CrashLoopBackOff

Revisa los logs del pod para ver el error específico:
```bash
kubectl logs -n mlops-proyecto3 deployment/airflow-webserver
kubectl logs -n mlops-proyecto3 deployment/airflow-scheduler
```

Errores comunes:
- **"No module named 'mlflow'"**: La imagen no se construyó correctamente. Reconstruye con `./build-image.sh`
- **"Database connection failed"**: Verifica que postgres-airflow esté corriendo

### Error: Base de datos no conecta

Verifica que los servicios de PostgreSQL estén corriendo:
```bash
kubectl get pods -n mlops-proyecto3 | grep postgres
```

Todos los pods de postgres deben estar en estado `Running` y `1/1 Ready`.

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
