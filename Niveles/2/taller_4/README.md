# Taller — Desarrollo en Contenedores (API + Jupyter + MLFlow + Docker Compose + Mino)

Este proyecto tiene como objetivo implementar una arquitectura MLOps híbrida que permite desplegar y gestionar un flujo completo de Machine Learning utilizando MLflow como herramienta central de tracking y versionado de modelos. La solución integra múltiples servicios mediante contenedores orquestados con Docker Compose, optimizando tanto el desarrollo local como el despliegue en entornos productivos.

### Estructura de Archivos

📁 taller4/
├── 📁 api/
│ ├── Dockerfile
│ └── app.py
├── 📁 mlflow/
│ ├── Dockerfile
│ ├── data_ingestion.py
│ ├── docker-compose.yml
│ ├── docker-compose.yml.bak
└── README.md

### Servios

| Servicio          | Imagen / Build                  | Puerto(s)      | Descripción                                                              |
| ----------------- | ------------------------------- | -------------- | ------------------------------------------------------------------------ |
| **MinIO**         | `quay.io/minio/minio:latest`    | `9000`, `9001` | Almacenamiento de artefactos S3-compatible.                              |
| **MySQL**         | `mysql:8.0`                     | `3306`         | Almacén de datos de entrenamiento y aplicación. Backend store de MLflow. |
| **MLflow**        | `./mlflow/Dockerfile`           | `5000`         | Tracking y registro de modelos. Integrado con MinIO y MySQL.             |
| **Jupyter**       | `jupyter/scipy-notebook:latest` | `8888`         | Entorno de desarrollo con conexión a MLflow, MinIO y MySQL.              |
| **API (FastAPI)** | `./api/Dockerfile`              | `8000`         | Servicio de inferencia que consume modelos del MLflow Registry.          |


Levantar los servicios que esten funcionales:
### Imagen 1 docker compose

### Volúmenes

| Volumen      | Descripción                                  |
| ------------ | -------------------------------------------- |
| `minio_data` | Persiste los datos almacenados en MinIO      |
| `mysql_data` | Persiste los datos de la base de datos MySQL |


### Dependencias

* MLflow depende de que MinIO y MySQL estén activos.
* Jupyter y API requieren de MLflow para funcionar correctamente.
* Todos los servicios comparten credenciales de acceso a MinIO

### Ejecución

#### Script de Ingesta de Datos

Archivo: mlflow/data_ingestion.py

El script se ejecuta fuera de los contenedores para insertar datos en MySQL. Crea automáticamente la base penguins_db, carga el dataset y realiza preprocesamiento básico.

Puedes ejecutarlo así:
export DATA_DB_URI="mysql+pymysql://mlflow_user:mlflow_pass@localhost:3306/penguins_db"
python mlflow/data_ingestion.py

#### API FastAPI para Inferencia

Archivo: api/app.py
Sirve el modelo en producción usando mlflow.pyfunc, con validación de esquema y coerción de tipos. Endpoints disponibles:

* GET /health
* GET /input-schema
* POST /predict
* POST /reload-model

#### Entrenamiento del Modelo

Abre Jupyter:
http://localhost:8888/?token=valentasecret

* Ejecuta el notebook en la carpeta notebooks/ para:
* Cargar los datos
* Entrenar modelos con GridSearch
* Registrar el mejor modelo en MLflow
* Promoverlo a Producción

### IMAGEN MLFLOW

#### Prueba de Inferencia

Con el API de FastAPI corriendo:
Consulta el esquema:
curl http://localhost:8000/input-schema

Realiza posteriormente una predicción:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'

