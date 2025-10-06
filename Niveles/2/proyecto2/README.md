# MLOps Proyecto 2 - Sistema de Predicción de Cobertura Forestal
## Grupo 6 - Pontificia Universidad Javeriana

## 📋 Descripción del Proyecto

Sistema completo de MLOps que implementa un pipeline de entrenamiento y predicción para clasificación de tipos de cobertura forestal (Forest Cover Type). El sistema utiliza:

- **Airflow**: Orquestación de pipelines de entrenamiento
- **MLflow**: Registro de experimentos y modelos
- **MinIO**: Almacenamiento de artefactos
- **MySQL**: Base de datos de metadata
- **PostgreSQL**: Base de datos de los batches
- **FastAPI**: API de inferencia
- **Streamlit**: Interfaz gráfica (BONO)

## 🏗️ Arquitectura

```
┌─────────────────────────────────────────────────────────┐
│                  Docker Compose                         │
│                                                         │
│  ┌──────────┐  ┌─────────┐  ┌────────┐  ┌──────────┐  │
│  │ Airflow  │→ │ MLflow  │→ │ MinIO  │  │  MySQL   │  │
│  │   DAG    │  │Tracking │  │ S3     │  │&& PostgreSQL│  │
│  └────┬─────┘  └────┬────┘  └────────┘  └──────────┘  │
│       │             │                                   │
│       ↓             ↓                                   │
│  ┌──────────┐  ┌─────────────┐                        │
│  │Data API  │  │  FastAPI    │← Puerto 8989           │
│  │External  │  │  Inference  │                        │
│  └──────────┘  └─────────────┘                        │
│                                                         │
│  ┌──────────────────┐                                  │
│  │   Streamlit UI   │← Puerto 8503 (BONO)             │
│  └──────────────────┘                                  │
└─────────────────────────────────────────────────────────┘
```

## 📁 Estructura del Proyecto

```
Proyecto2/
├── docker-compose.yml         # Orquestación de servicios
├── .env                       # Variables de entorno
├── README.md                  # Este archivo
│
├── airflow/                   # Servicio Airflow
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       └── training_pipeline.py
│
├── mlflow/                    # Servicio MLflow
│   └── Dockerfile
│
├── inference_api/             # API de Inferencia
│   ├── Dockerfile
│   ├── main.py
│   └── requirements.txt
│
├── data_api/                  # API de Datos (del profesor)
│   └── (código del repositorio del profesor)
│
└── ui/                        # Interfaz Gráfica (BONO)
    ├── Dockerfile
    ├── app.py
    └── requirements.txt
```

## 🚀 Instrucciones de Instalación

### Prerequisitos

- Docker Desktop instalado y corriendo
- Docker Compose v2.0+
- Git
- Al menos 8GB de RAM disponible
- 20GB de espacio en disco

### Paso 1: Clonar el Repositorio

```bash
# Clonar el repositorio del grupo
git clone https://github.com/Saracar2021/MLOPS_PUJ_Grupo6.git
cd MLOPS_PUJ_Grupo6

# Crear carpeta del proyecto
mkdir -p Proyecto2
cd Proyecto2
```

### Paso 2: Clonar la API de Datos del Profesor

```bash
# Clonar la API de datos en la carpeta data_api
git clone https://github.com/CristianDiazAlvarez/MLOPS_PUJ.git temp_mlops
cp -r temp_mlops/Niveles/2/P2/* data_api/
rm -rf temp_mlops
```

### Paso 3: Crear la Estructura de Carpetas

```bash
# Crear todas las carpetas necesarias
mkdir -p airflow/dags airflow/logs airflow/plugins
mkdir -p mlflow
mkdir -p inference_api
mkdir -p ui
```

### Paso 4: Copiar los Archivos

Copia todos los archivos proporcionados en los artifacts a sus respectivas ubicaciones según la estructura mostrada arriba.

### Paso 5: Configurar Variables de Entorno

Crea el archivo `.env` en la raíz del proyecto:

```bash
# MySQL
MYSQL_ROOT_PASSWORD=rootpassword
MYSQL_DATABASE=mlflow_db
MYSQL_USER=mlflow_user
MYSQL_PASSWORD=mlflow_password

# MinIO
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=minioadmin
MINIO_BUCKET=mlflow

# MLflow
MLFLOW_TRACKING_URI=http://mlflow:5000
MLFLOW_S3_ENDPOINT_URL=http://minio:9000

# API Externa (modificar según necesidad)
DATA_API_URL=http://data_api:8080
GROUP_NUMBER=6

# Airflow
AIRFLOW_UID=50000
```

### Paso 6: Construir y Levantar los Servicios

```bash
# Construir todas las imágenes
docker-compose build

# Levantar todos los servicios
docker-compose up -d

# Ver logs de todos los servicios
docker-compose logs -f

# Ver logs de un servicio específico
docker-compose logs -f mlflow
```

### Paso 7: Verificar que los Servicios Estén Corriendo

```bash
# Ver estado de los contenedores
docker-compose ps

# Todos los servicios deben mostrar estado "Up" o "healthy"
```

## 🌐 Acceso a los Servicios

Una vez que todos los servicios estén corriendo:

| Servicio | URL | Credenciales |
|----------|-----|--------------|
| **Airflow** | http://localhost:8080 | usuario: `admin` / password: `admin` |
| **MLflow** | http://localhost:5000 | Sin autenticación |
| **MinIO Console** | http://localhost:9001 | usuario: `minioadmin` / password: `minioadmin` |
| **Inference API** | http://localhost:8989 | Sin autenticación |
| **API Docs** | http://localhost:8989/docs | Documentación interactiva |
| **Streamlit UI (BONO)** | http://localhost:8503 | Sin autenticación |

## 📊 Flujo de Trabajo

### 1. Entrenamiento de Modelos

1. Accede a Airflow: http://localhost:8080
2. Busca el DAG: `forest_cover_training_pipeline`
3. Activa el DAG (toggle en ON)
4. Ejecuta manualmente: Click en "Trigger DAG"
5. Monitorea la ejecución en tiempo real

El DAG ejecutará:
- ✅ Obtención de datos de la API externa
- ✅ Preprocesamiento de datos
- ✅ Entrenamiento de 3 modelos (Logistic Regression, Random Forest, Gradient Boosting)
- ✅ Registro en MLflow con métricas y artefactos

### 2. Revisión de Experimentos

1. Accede a MLflow: http://localhost:5000
2. Selecciona el experimento: `forest_cover_classification`
3. Compara métricas entre modelos
4. Revisa artefactos y parámetros

### 3. Realizar Predicciones

#### Opción A: Usando la API Directamente

```bash
# Listar modelos disponibles
curl http://localhost:8989/models

# Cargar un modelo específico
curl -X POST "http://localhost:8989/models/forest_cover_random_forest/load?stage=Production"

# Realizar predicción
curl -X POST "http://localhost:8989/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Elevation": 2596,
    "Aspect": 51,
    "Slope": 3,
    "Horizontal_Distance_To_Hydrology": 258,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 510,
    "Hillshade_9am": 221,
    "Hillshade_Noon": 232,
    "Hillshade_3pm": 148,
    "Horizontal_Distance_To_Fire_Points": 6279
  }'
```


### ✅ Requerimientos Obligatorios

- [x] Docker Compose con todos los servicios
- [x] Airflow para orquestación
- [x] MLflow con registro de experimentos
- [x] MinIO como object storage
- [x] MySQL para metadata
- [x] API de inferencia con FastAPI (puerto 8989)
- [x] Entrenamiento de múltiples modelos
- [x] Pipeline automatizado completo
- [x] Documentación completa

### Comandos utiles

```bash
# Detener todos los servicios
docker-compose down

# Detener y eliminar volúmenes (CUIDADO: elimina datos)
docker-compose down -v

# Reiniciar un servicio específico
docker-compose restart airflow-webserver

# Ver logs en tiempo real
docker-compose logs -f

# Ejecutar comando dentro de un contenedor
docker-compose exec airflow-webserver bash

# Ver recursos utilizados
docker stats

# Limpiar todo (imágenes, contenedores, volúmenes)
docker system prune -a --volumes
```

## 🔧 Troubleshooting

### Error: Puerto ya en uso

```bash
# Ver qué proceso está usando el puerto
lsof -i :8080  # o el puerto que esté en conflicto

# Matar el proceso
kill -9 <PID>
```

### Error: Contenedor no levanta

```bash
# Ver logs específicos
docker-compose logs nombre_servicio

# Verificar configuración
docker-compose config
```

### Error: MLflow no conecta con MinIO

```bash
# Verificar que MinIO esté corriendo
docker-compose ps minio

# Reiniciar MLflow
docker-compose restart mlflow
```

### Error: Airflow no puede conectar con MLflow

```bash
# Verificar red de Docker
docker network ls
docker network inspect proyecto2_mlops_network

# Reiniciar ambos servicios
docker-compose restart airflow-scheduler airflow-webserver mlflow
```

## 📝 Notas Importantes

1. **Primera ejecución**: La primera vez puede tardar 5-10 minutos en que todos los servicios estén listos

2. **Recursos**: Asegúrate de asignar suficientes recursos a Docker (mínimo 8GB RAM)

3. **API de Datos**: Si la API del profesor no está disponible, el código de la API está en `data_api/` para ejecutarla localmente

4. **Persistencia**: Los datos se mantienen en volúmenes de Docker. Para limpiar completamente, usa `docker-compose down -v`

5. **Desarrollo**: Para hacer cambios en el código, solo necesitas reiniciar el servicio específico:
   ```bash
   docker-compose restart inference_api
   ```

## 📚 Dataset

El proyecto utiliza el dataset **Forest Cover Type** que predice el tipo de cobertura forestal basado en variables cartográficas.

### Tipos de Cobertura (1-7):
1. Spruce/Fir
2. Lodgepole Pine
3. Ponderosa Pine
4. Cottonwood/Willow
5. Aspen
6. Douglas-fir
7. Krummholz

### Variables de Entrada:
- Elevation (metros)
- Aspect (grados azimuth)
- Slope (grados)
- Distancias a características geográficas
- Índices de sombra

## 👥 Equipo

**Grupo 6**  
Pontificia Universidad Javeriana  
Materia: Operaciones de Machine Learning  
Proyecto 2 - Nivel 2

## 📄 Licencia

Este proyecto es para fines académicos.

## 🆘 Soporte

Si encuentras problemas:
1. Revisa la sección de Troubleshooting
2. Verifica los logs: `docker-compose logs -f`
3. Consulta la documentación de cada servicio
4. Contacta al equipo de desarrollo

---

**¡Sistema MLOps completamente funcional! 🚀**
