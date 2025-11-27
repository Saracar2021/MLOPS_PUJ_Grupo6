from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import pandas as pd
import json
import psycopg2
from psycopg2.extras import execute_values
import os
import logging

logger = logging.getLogger(__name__)

default_args = {
    'owner': 'grupo9',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 25),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# Configuración
API_BASE = "http://10.43.100.103:8000"
GROUP_NUMBER = 9
DAY = "Wednesday"
BASELINE_SAMPLE = 145000  # Primera petición: completa
INCREMENTAL_SAMPLE = 10000  # Peticiones siguientes: sample

def get_db_connection():
    """Conectar a PostgreSQL RAW database"""
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database='raw_data',
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )

def init_raw_database(**context):
    """Crear tabla RAW si no existe"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id SERIAL PRIMARY KEY,
            request_number INT NOT NULL,
            received_at TIMESTAMP DEFAULT NOW(),
            data JSONB NOT NULL,
            sample_size INT,
            used_in_training BOOLEAN DEFAULT FALSE
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info("✅ Tabla raw_data inicializada")

def get_next_request_number(**context):
    """Obtener número de petición actual"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COALESCE(MAX(request_number), 0) + 1 FROM raw_data")
    request_num = cur.fetchone()[0]
    
    cur.close()
    conn.close()
    
    logger.info(f"📊 Próxima petición: #{request_num}")
    context['task_instance'].xcom_push(key='request_number', value=request_num)
    return request_num

def fetch_from_api(**context):
    """Hacer petición a API externa"""
    request_num = context['task_instance'].xcom_pull(key='request_number')
    
    url = f"{API_BASE}/data"
    params = {"group_number": GROUP_NUMBER, "day": DAY}
    
    logger.info(f"🌐 Petición #{request_num} a {url}")
    logger.info(f"   Parámetros: {params}")
    
    try:
        response = requests.get(url, params=params, timeout=60)
        
        if response.status_code == 200:
            data = response.json()
            
            if isinstance(data, dict) and 'data' in data:
                records = data['data']
            elif isinstance(data, list):
                records = data
            else:
                raise ValueError(f"Formato inesperado: {type(data)}")
            
            original_count = len(records)
            logger.info(f"✅ Recibidos {original_count} registros")
            
            context['task_instance'].xcom_push(key='api_data', value=records)
            context['task_instance'].xcom_push(key='original_count', value=original_count)
            
            return original_count
            
        elif response.status_code == 404:
            logger.warning("⚠️ API retornó 404: No hay más datos disponibles")
            return 0
        else:
            raise Exception(f"Error API: {response.status_code} - {response.text}")
            
    except requests.exceptions.Timeout:
        logger.error("⏱️ Timeout (60s) - API no responde")
        raise
    except Exception as e:
        logger.error(f"❌ Error en petición: {str(e)}")
        raise

def sample_data(**context):
    """Aplicar sampling inteligente"""
    request_num = context['task_instance'].xcom_pull(key='request_number')
    records = context['task_instance'].xcom_pull(key='api_data')
    original_count = context['task_instance'].xcom_pull(key='original_count')
    
    if request_num == 1:
        # Primera petición: mantener tamaño baseline
        sample_size = min(BASELINE_SAMPLE, original_count)
        logger.info(f"📊 Petición #1 (BASELINE): {original_count} → {sample_size} registros")
    else:
        # Peticiones incrementales: sampling
        sample_size = min(INCREMENTAL_SAMPLE, original_count)
        logger.info(f"📊 Petición #{request_num} (INCREMENTAL): {original_count} → {sample_size} registros")
    
    # Aplicar sampling si es necesario
    if original_count > sample_size:
        df = pd.DataFrame(records)
        df_sample = df.sample(n=sample_size, random_state=42)
        sampled_records = df_sample.to_dict('records')
        logger.info(f"🔄 Sampling aplicado: {len(sampled_records)} registros seleccionados")
    else:
        sampled_records = records
        logger.info(f"✅ Sin sampling necesario")
    
    context['task_instance'].xcom_push(key='sampled_data', value=sampled_records)
    context['task_instance'].xcom_push(key='sample_size', value=sample_size)

def store_raw_data(**context):
    """Guardar datos en PostgreSQL RAW"""
    request_num = context['task_instance'].xcom_pull(key='request_number')
    records = context['task_instance'].xcom_pull(key='sampled_data')
    sample_size = context['task_instance'].xcom_pull(key='sample_size')
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    # Preparar datos para inserción
    values = [(request_num, json.dumps(record), sample_size) for record in records]
    
    execute_values(
        cur,
        """
        INSERT INTO raw_data (request_number, data, sample_size)
        VALUES %s
        """,
        values,
        template="(%s, %s::jsonb, %s)"
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info(f"💾 Almacenados {len(records)} registros en raw_data")
    logger.info(f"✅ Petición #{request_num} completada exitosamente")

with DAG(
    'data_ingestion',
    default_args=default_args,
    description='Ingesta de datos desde API externa con sampling inteligente',
    schedule_interval=None,  # Trigger manual
    catchup=False,
    tags=['ingestion', 'grupo9']
) as dag:
    
    init_db = PythonOperator(
        task_id='init_raw_database',
        python_callable=init_raw_database,
    )
    
    get_request_num = PythonOperator(
        task_id='get_next_request_number',
        python_callable=get_next_request_number,
    )
    
    fetch_api = PythonOperator(
        task_id='fetch_from_api',
        python_callable=fetch_from_api,
    )
    
    sample = PythonOperator(
        task_id='sample_data',
        python_callable=sample_data,
    )
    
    store_raw = PythonOperator(
        task_id='store_raw_data',
        python_callable=store_raw_data,
    )
    
    init_db >> get_request_num >> fetch_api >> sample >> store_raw
