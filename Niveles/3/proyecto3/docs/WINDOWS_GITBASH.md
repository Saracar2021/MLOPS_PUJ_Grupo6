# INSTRUCCIONES PARA WINDOWS (Git Bash)

## INSTALACIÓN PASO A PASO

### 1. Descargar el Proyecto

```bash
# Opción A: Desde Claude
# El proyecto está disponible en: proyecto3_mlops.zip
# Descomprimir en una carpeta (ej: C:\Users\TuUsuario\mlops\proyecto3_mlops)

# Opción B: Desde Git (si está en GitHub)
git clone <URL_REPO>
cd proyecto3_mlops
```

### 2. Instalar Docker Desktop

1. Descargar desde: https://www.docker.com/products/docker-desktop
2. Instalar y **REINICIAR** el sistema
3. Iniciar Docker Desktop
4. Verificar instalación:

```bash
docker --version
# Resultado esperado: Docker version 24.x.x

docker ps
# Resultado esperado: Lista vacía (sin errores)
```

### 3. Instalar Minikube

**Opción A - Instalador Windows (Recomendado)**:
1. Descargar: https://github.com/kubernetes/minikube/releases/latest/download/minikube-installer.exe
2. Ejecutar instalador
3. Verificar en Git Bash:

```bash
minikube version
# Resultado esperado: minikube version: v1.32.x
```

**Opción B - Manual con Git Bash**:
```bash
# Crear carpeta para binarios
mkdir -p $HOME/bin

# Descargar Minikube
curl -Lo minikube.exe https://github.com/kubernetes/minikube/releases/latest/download/minikube-windows-amd64.exe
mv minikube.exe $HOME/bin/

# Agregar al PATH
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
source ~/.bashrc

# Verificar
minikube version
```

### 4. Instalar kubectl

**Opción A - Con Docker Desktop**:
Docker Desktop incluye kubectl. Verificar:
```bash
kubectl version --client
```

**Opción B - Manual**:
```bash
curl -Lo kubectl.exe https://dl.k8s.io/release/v1.28.0/bin/windows/amd64/kubectl.exe
mv kubectl.exe $HOME/bin/
kubectl version --client
```

### 5. Iniciar Minikube

```bash
# IMPORTANTE: Abrir Git Bash como ADMINISTRADOR

# Iniciar Minikube (usar 16GB RAM de los 32GB disponibles)
minikube start --memory=16384 --cpus=6 --driver=docker

# Esto tomará 3-5 minutos la primera vez

# Verificar que está corriendo
minikube status

# Resultado esperado:
# minikube
# type: Control Plane
# host: Running
# kubelet: Running
# apiserver: Running
# kubeconfig: Configured
```

## DESPLEGAR EL PROYECTO

### Opción 1: Script Automatizado (Recomendado)

```bash
# Navegar a la carpeta del proyecto
cd /c/Users/TuUsuario/mlops/proyecto3_mlops

# IMPORTANTE: Convertir line endings si es necesario
dos2unix deploy.sh || sed -i 's/\r$//' deploy.sh

# Hacer ejecutable
chmod +x deploy.sh

# Ejecutar deployment
./deploy.sh

# Este script desplegará todo automáticamente (~10-15 minutos)
```

### Opción 2: Manual (Paso a Paso)

```bash
# 1. Crear namespace
kubectl create namespace mlops-proyecto3
kubectl config set-context --current --namespace=mlops-proyecto3

# 2. Desplegar bases de datos
kubectl apply -f k8s/postgres-raw/
kubectl apply -f k8s/postgres-clean/
kubectl apply -f k8s/postgres-mlflow/

# Esperar que estén ready (2-3 minutos)
kubectl wait --for=condition=ready pod -l app=postgres-raw --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres-clean --timeout=300s
kubectl wait --for=condition=ready pod -l app=postgres-mlflow --timeout=300s

# 3. Desplegar MinIO
kubectl apply -f k8s/minio/
kubectl wait --for=condition=ready pod -l app=minio --timeout=300s

# 4. Desplegar MLflow
kubectl apply -f k8s/mlflow/
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s

# 5. Desplegar Airflow
kubectl apply -f k8s/airflow/
sleep 30
kubectl wait --for=condition=ready pod -l app=airflow-webserver --timeout=300s

# 6. Desplegar FastAPI
kubectl apply -f k8s/fastapi/
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=300s

# 7. Desplegar Streamlit
kubectl apply -f k8s/streamlit/
kubectl wait --for=condition=ready pod -l app=streamlit --timeout=300s

# 8. Desplegar Observabilidad
kubectl apply -f k8s/prometheus/
kubectl apply -f k8s/grafana/
kubectl wait --for=condition=ready pod -l app=prometheus --timeout=300s
kubectl wait --for=condition=ready pod -l app=grafana --timeout=300s
```

## ACCEDER A LOS SERVICIOS

### Ver URLs de Todos los Servicios

```bash
# Lista de servicios
minikube service list -n mlops-proyecto3
```

### Acceder a cada servicio

```bash
# Airflow (abrirá en navegador)
minikube service airflow-webserver -n mlops-proyecto3
# Credenciales: admin / admin

# MLflow
minikube service mlflow -n mlops-proyecto3

# Streamlit (interfaz principal)
minikube service streamlit -n mlops-proyecto3

# FastAPI
minikube service fastapi -n mlops-proyecto3

# Grafana
minikube service grafana -n mlops-proyecto3
# Credenciales: admin / admin
```

### Alternativa: Obtener URLs sin abrir navegador

```bash
minikube service airflow-webserver -n mlops-proyecto3 --url
# Copiar URL y pegar en navegador
```

## TROUBLESHOOTING WINDOWS/GIT BASH

### Problema: "minikube command not found"

```bash
# Verificar PATH
echo $PATH | grep -o "$HOME/bin"

# Si no aparece, agregar:
echo 'export PATH=$PATH:$HOME/bin' >> ~/.bashrc
source ~/.bashrc
```

### Problema: "Cannot connect to Docker"

```bash
# Verificar que Docker Desktop está corriendo
docker ps

# Si falla, abrir Docker Desktop manualmente desde el menú de inicio
```

### Problema: Minikube no inicia

```bash
# Limpiar instalación anterior
minikube delete

# Reiniciar con configuración específica
minikube start --memory=16384 --cpus=6 --driver=docker --force

# Si persiste, verificar virtualización:
# 1. BIOS debe tener Hyper-V habilitado
# 2. En Windows, ejecutar: bcdedit /set hypervisorlaunchtype auto
```

### Problema: "permission denied" en scripts

```bash
# Convertir line endings de Windows a Unix
sed -i 's/\r$//' deploy.sh

# Dar permisos de ejecución
chmod +x deploy.sh
```

### Problema: Servicios no accesibles

```bash
# Opción 1: Usar minikube tunnel (requiere admin)
minikube tunnel

# Opción 2: Port-forward específico
kubectl port-forward service/streamlit 8501:8501 -n mlops-proyecto3
# Acceder a http://localhost:8501
```

### Problema: Pods en CrashLoopBackOff

```bash
# Ver logs del pod
kubectl logs <pod-name> -n mlops-proyecto3

# Ver eventos
kubectl describe pod <pod-name> -n mlops-proyecto3

# Reiniciar deployment
kubectl rollout restart deployment/<name> -n mlops-proyecto3
```

## COMANDOS ÚTILES EN GIT BASH

```bash
# Ver todos los pods
kubectl get pods -n mlops-proyecto3

# Ver logs en tiempo real
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3

# Ejecutar comando en pod
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- bash

# Ver uso de recursos
kubectl top nodes
kubectl top pods -n mlops-proyecto3

# Limpiar todo
kubectl delete namespace mlops-proyecto3
minikube stop
```

## EJECUCIÓN DEL PIPELINE COMPLETO

### 1. Activar DAGs en Airflow

```bash
# Acceder a Airflow UI
minikube service airflow-webserver -n mlops-proyecto3

# En la UI:
# 1. Toggle ON para cada DAG:
#    - data_ingestion_batch
#    - data_processing_pipeline
#    - model_training_pipeline
#    - model_promotion_pipeline

# 2. Esperar ejecución automática o trigger manual
```

### 2. Monitorear Progreso

```bash
# Ver logs de scheduler
kubectl logs -f deployment/airflow-scheduler -n mlops-proyecto3

# Ver datos cargados
kubectl exec -it deployment/postgres-raw -n mlops-proyecto3 -- \
  psql -U mlops_user -d raw_db -c "SELECT batch_id, COUNT(*) FROM raw_data GROUP BY batch_id;"
```

### 3. Verificar Modelos en MLflow

```bash
# Acceder a MLflow UI
minikube service mlflow -n mlops-proyecto3

# Verificar:
# - Experimento "diabetes_readmission"
# - 3 runs registrados
# - Model Registry con modelo en "Production"
```

### 4. Probar Predicciones

```bash
# Opción A: Streamlit UI
minikube service streamlit -n mlops-proyecto3

# Opción B: API directa
FASTAPI_URL=$(minikube service fastapi -n mlops-proyecto3 --url)
curl -X POST "$FASTAPI_URL/predict" -H "Content-Type: application/json" -d '{...}'
```

### 5. Load Testing con Locust

```bash
# Port-forward Locust
kubectl port-forward service/locust 8089:8089 -n mlops-proyecto3

# Acceder a http://localhost:8089
# Configurar 100 usuarios, spawn rate 10
```

### 6. Ver Métricas en Grafana

```bash
# Acceder a Grafana
minikube service grafana -n mlops-proyecto3
# admin / admin

# Agregar Prometheus data source:
# URL: http://prometheus:9090
```

## DETENER Y LIMPIAR

```bash
# Detener Minikube (conserva datos)
minikube stop

# Eliminar solo el proyecto (conserva Minikube)
kubectl delete namespace mlops-proyecto3

# Reset completo (elimina todo)
minikube delete
```

## TIPS PARA WINDOWS

1. **Siempre usar Git Bash como Administrador** para operaciones de Minikube
2. **Docker Desktop debe estar corriendo** antes de usar Minikube
3. **Cerrar antivirus/firewall temporalmente** si hay problemas de red
4. **Aumentar memoria de WSL2** si tienes menos de 32GB RAM
5. **Usar rutas Unix** en Git Bash: `/c/Users/...` en lugar de `C:\Users\...`

## RECURSOS ADICIONALES

- [Documentación Minikube](https://minikube.sigs.k8s.io/docs/)
- [Kubectl Cheat Sheet](https://kubernetes.io/docs/reference/kubectl/cheatsheet/)
- [Docker Desktop para Windows](https://docs.docker.com/desktop/windows/install/)
- [Git Bash](https://gitforwindows.org/)

## CONTACTO Y SOPORTE

Si encuentras problemas:
1. Revisa la sección TROUBLESHOOTING arriba
2. Consulta `docs/QUICK_REFERENCE.md` para comandos comunes
3. Revisa logs con: `kubectl logs <pod-name> -n mlops-proyecto3`
