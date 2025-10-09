# Taller — Desarrollo en Contenedores (API + Jupyter + MLFlow + Docker Compose + Mino)

Este proyecto tiene como objetivo implementar una arquitectura MLOps híbrida que permite desplegar y gestionar un flujo completo de Machine Learning utilizando MLflow como herramienta central de tracking y versionado de modelos. La solución integra múltiples servicios mediante contenedores orquestados con Docker Compose, optimizando tanto el desarrollo local como el despliegue en entornos productivos.


### Estructura de Archivos

```
taller4
├── API/                   
│   ├── Dockerfile
│   ├── app.py
│   
├── mlflow/                    # Servicio MLflow
│   ├── Dockerfile
│   ├── data_ingestion.py
│   └── docker-compose.yml
│   └── docker-compose.yml.bak
└── README.md

```
### Servicios

| Servicio          | Imagen / Build                  | Puerto(s)      | Descripción                                                              |
| ----------------- | ------------------------------- | -------------- | ------------------------------------------------------------------------ |
| **MinIO**         | `quay.io/minio/minio:latest`    | `9000`, `9001` | Almacenamiento de artefactos S3-compatible.                              |
| **MySQL**         | `mysql:8.0`                     | `3306`         | Almacén de datos de entrenamiento y aplicación. Backend store de MLflow. |
| **MLflow**        | `./mlflow/Dockerfile`           | `5000`         | Tracking y registro de modelos. Integrado con MinIO y MySQL.             |
| **Jupyter**       | `jupyter/scipy-notebook:latest` | `8888`         | Entorno de desarrollo con conexión a MLflow, MinIO y MySQL.              |
| **API (FastAPI)** | `./api/Dockerfile`              | `8000`         | Servicio de inferencia que consume modelos del MLflow Registry.          |

<img width="697" height="577" alt="Contenedores 2025-10-03 a la(s) 7 56 18 p m" src="https://github.com/user-attachments/assets/2446d3fe-2933-4708-8da9-f8cce88be15e" />

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

<img width="1437" height="415" alt="MLflow 2025-10-03 a la(s) 6 52 55 p m" src="https://github.com/user-attachments/assets/8d0536f4-f9b8-4fd6-90c0-1bcc5f853258" />
<img width="1262" height="752" alt="Mlflow Exitoso 2025-10-03 a la(s) 6 53 13 p m" src="https://github.com/user-attachments/assets/abd65887-851a-425e-b022-bacb1ae8de11" />
<img width="1772" height="986" alt="imagen (5)" src="https://github.com/user-attachments/assets/cacf3e03-739d-41bd-ae1a-9b56aaf23692" />

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
<img width="1436" height="544" alt="Prediccion 2025-10-03 a la(s) 6 52 37 p m" src="https://github.com/user-attachments/assets/0390fc12-4bb3-435b-aea5-96aa4289dfc0" />


#### Prueba de Inferencia

Con el API de FastAPI corriendo:
Consulta el esquema:
curl http://localhost:8000/input-schema

Realiza posteriormente una predicción:
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"dataframe_split": {"columns": [...], "data": [[...]]}}'


