FROM apache/airflow:2.7.0-python3.9

USER root
# Instalar dependencias del sistema si son necesarias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

USER airflow

# Copiar requirements e instalar
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt

# Copiar DAGs a ubicación temporal
COPY dags/ /opt/airflow/dags-source/

WORKDIR /opt/airflow
