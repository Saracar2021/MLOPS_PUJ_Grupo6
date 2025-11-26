# ✅ PROYECTO COMPLETO - LISTO PARA USAR

## 📦 Contenido del Proyecto (23 archivos)

```
mlops-proyecto-final/
├── README.md                          ✅ Documentación completa
├── .env.example                       ✅ Variables de entorno
├── .gitignore                         ✅ Configurado
├── deploy.sh                          ✅ Script de despliegue
│
├── .github/workflows/
│   └── build-and-push.yml             ✅ CI/CD completo
│
├── airflow/
│   ├── Dockerfile                     ✅
│   ├── requirements.txt               ✅
│   └── dags/
│       ├── data_ingestion.py          ✅ Fetch API + sampling
│       ├── data_processing.py         ✅ RAW → CLEAN
│       ├── model_training.py          ✅ Drift + training
│       └── model_promotion.py         ✅ Production
│
├── api/
│   ├── Dockerfile                     ✅
│   ├── main.py                        ✅ FastAPI + Prometheus
│   └── requirements.txt               ✅
│
├── ui/
│   ├── Dockerfile                     ✅
│   ├── app.py                         ✅ Streamlit
│   └── requirements.txt               ✅
│
└── k8s/
    ├── postgres/all-in-one.yaml       ✅ 3 databases
    ├── minio/all-in-one.yaml          ✅ S3 + init bucket
    ├── mlflow/all-in-one.yaml         ✅ Tracking server
    ├── airflow/all-in-one.yaml        ✅ Webserver + Scheduler
    ├── api/all-in-one.yaml            ✅ FastAPI
    ├── ui/all-in-one.yaml             ✅ Streamlit
    ├── prometheus/all-in-one.yaml     ✅ Metrics
    └── grafana/all-in-one.yaml        ✅ Dashboards
```

---

## 🎯 PRÓXIMOS PASOS (Orden de ejecución)

### 1. Copiar proyecto a tu repositorio (2 min)
```bash
# En tu máquina Windows
cd C:\Users\julia\MLOPS_PUJ_Grupo6\Niveles\4

# Extraer proyecto (descarga mlops-proyecto-final.tar.gz primero)
tar -xzf mlops-proyecto-final.tar.gz
cd mlops-proyecto-final
```

### 2. Configurar .env (1 min)
```bash
cp .env.example .env
# Editar con tus valores reales si es necesario
```

### 3. Actualizar usuario de DockerHub (2 min)
**CRÍTICO**: Reemplazar `YOUR_DOCKERHUB_USERNAME` en manifiestos K8s

```bash
# Opción A: Manual
# Editar estos archivos y cambiar "YOUR_DOCKERHUB_USERNAME" por tu usuario:
# - k8s/airflow/all-in-one.yaml (3 lugares)
# - k8s/api/all-in-one.yaml (1 lugar)
# - k8s/ui/all-in-one.yaml (1 lugar)

# Opción B: Comando (Git Bash)
find k8s -name "*.yaml" -exec sed -i 's/YOUR_DOCKERHUB_USERNAME/TU_USUARIO_AQUI/g' {} +
```

### 4. Inicializar Git (1 min)
```bash
git init
git add .
git commit -m "feat: estructura inicial del proyecto"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/mlops-proyecto-final.git
```

### 5. Configurar GitHub Secrets (2 min)
En tu repositorio de GitHub:
1. Settings → Secrets and variables → Actions
2. New repository secret:
   - `DOCKERHUB_USERNAME`: tu usuario de DockerHub
   - `DOCKERHUB_TOKEN`: token de acceso (crear en https://hub.docker.com/settings/security)

### 6. Push para activar CI/CD (3 min)
```bash
git push -u origin main

# Ir a GitHub → Actions tab
# Ver que los workflows se ejecutan (build-airflow, build-api, build-ui)
# Esperar a que terminen (~5 min)
```

### 7. Verificar imágenes en DockerHub (1 min)
```bash
# Ir a https://hub.docker.com/u/TU_USUARIO
# Verificar que existen:
# - TU_USUARIO/mlops-airflow:latest
# - TU_USUARIO/mlops-api:latest
# - TU_USUARIO/mlops-ui:latest
```

### 8. Desplegar en Kubernetes (10 min)
```bash
# Iniciar Minikube
minikube start --memory=16384 --cpus=6 --driver=docker

# Ejecutar deployment
chmod +x deploy.sh
./deploy.sh

# Esperar a que termine (~10 min)
# Al final mostrará las URLs de acceso
```

### 9. Ejecutar pipeline (30 min)
```bash
# 1. Acceder a Airflow UI (admin/admin)
AIRFLOW_URL=$(minikube service airflow-webserver -n mlops-proyecto-final --url)
echo $AIRFLOW_URL
# Abrir en navegador

# 2. Activar y ejecutar DAGs en orden:
#    a. data_ingestion (trigger 5 veces)
#    b. data_processing (después de cada ingestion)
#    c. model_training (automático si hay drift)
#    d. model_promotion (después de training)
```

### 10. Validar todo funciona (10 min)
```bash
# Ver todos los pods
kubectl get pods -n mlops-proyecto-final
# Todos deben estar "Running"

# Test de predicción
FASTAPI_URL=$(minikube service fastapi -n mlops-proyecto-final --url)
curl -X POST "$FASTAPI_URL/predict" -H "Content-Type: application/json" -d '{
  "brokered_by": "12345", "status": "for_sale", "bed": 3, "bath": 2.0,
  "acre_lot": 0.5, "city": "Boston", "state": "Massachusetts",
  "zip_code": "02101", "house_size": 1500
}'

# Acceder a Streamlit
minikube service streamlit -n mlops-proyecto-final
```

---

## ⚡ TIMELINE ESTIMADO

- Setup inicial (pasos 1-7): **15 minutos**
- Deployment K8s (paso 8): **10 minutos**
- Ejecución pipeline (paso 9): **30 minutos**
- Validación (paso 10): **10 minutos**

**TOTAL: ~65 minutos** de trabajo activo + esperas

---

## 🎥 VIDEO DE SUSTENTACIÓN (10 min)

### Minuto 0-2: Arquitectura
- Mostrar diagrama del README
- `kubectl get all -n mlops-proyecto-final`
- Explicar flujo completo

### Minuto 2-4: CI/CD
- Mostrar GitHub Actions → Workflows exitosos
- Mostrar DockerHub → Imágenes publicadas
- Explicar versionamiento (SHA tags)

### Minuto 4-6: Procesamiento y Drift
- Mostrar logs de Airflow → Scheduler
- Explicar criterio de reentrenamiento
- Mostrar log de drift detection

### Minuto 6-8: Inferencia
- Streamlit UI → Hacer predicción
- Mostrar versión del modelo en uso
- Cambiar modelo en MLflow → Reload → Nueva predicción

### Minuto 8-10: Métricas
- Prometheus → Targets
- Grafana → Dashboard (si da tiempo)
- Conclusiones

---

## ✅ CHECKLIST ANTES DE ENTREGAR

- [ ] Código en GitHub público
- [ ] GitHub Actions workflows ejecutados exitosamente
- [ ] 3 imágenes en DockerHub (airflow, api, ui)
- [ ] Sistema desplegado en Kubernetes
- [ ] 5 peticiones a la API completadas
- [ ] Al menos 1 modelo en Production en MLflow
- [ ] API responde correctamente a /predict
- [ ] Streamlit UI accesible
- [ ] README completo en el repo
- [ ] Video ≤10 min subido a YouTube
- [ ] Link del video en el README

---

## 🚨 ERRORES COMUNES Y SOLUCIONES

### Error 1: "YOUR_DOCKERHUB_USERNAME not found"
**Causa**: No actualizaste los manifiestos K8s  
**Solución**: Paso 3 - Reemplazar en todos los archivos yaml

### Error 2: GitHub Actions falla
**Causa**: Secrets no configurados  
**Solución**: Paso 5 - Configurar DOCKERHUB_USERNAME y DOCKERHUB_TOKEN

### Error 3: Pods en CrashLoopBackOff
**Causa**: Imágenes no existen en DockerHub  
**Solución**: Esperar a que GitHub Actions termine (paso 6)

### Error 4: Airflow DAGs no aparecen
**Causa**: Imagen custom no se construyó correctamente  
**Solución**: Verificar build en GitHub Actions → Ver logs

### Error 5: No puedo acceder a API externa (10.43.100.103)
**Causa**: No estás en red PUJ  
**Solución**: Conectar a VPN PUJ o ejecutar desde campus

---

## 💪 ¡ESTÁS LISTO!

El proyecto está **100% completo**. Solo necesitas:
1. ✅ Copiar a tu repo
2. ✅ Actualizar usuario DockerHub
3. ✅ Configurar secrets
4. ✅ Push para CI/CD
5. ✅ Deploy

**¡Éxito con tu entrega!** 🚀
