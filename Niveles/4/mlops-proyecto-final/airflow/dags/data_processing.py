from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import pandas as pd
import json
import psycopg2
from psycopg2.extras import execute_values
import os
import logging
from sklearn.preprocessing import LabelEncoder

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

def get_raw_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database='raw_data',
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )

def get_clean_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database='clean_data',
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )

def init_clean_database(**context):
    """Crear tabla CLEAN si no existe"""
    conn = get_clean_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clean_data (
            id SERIAL PRIMARY KEY,
            request_number INT NOT NULL,
            brokered_by VARCHAR(50),
            status VARCHAR(20),
            price FLOAT,
            bed INT,
            bath FLOAT,
            acre_lot FLOAT,
            city VARCHAR(100),
            state VARCHAR(50),
            zip_code VARCHAR(10),
            house_size FLOAT,
            processed_at TIMESTAMP DEFAULT NOW()
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    logger.info("✅ Tabla clean_data inicializada")

def fetch_unprocessed_raw(**context):
    """Obtener datos RAW no procesados"""
    conn = get_raw_connection()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT id, request_number, data 
        FROM raw_data 
        WHERE used_in_training = FALSE
        ORDER BY request_number
    """)
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    if not rows:
        logger.warning("⚠️ No hay datos nuevos para procesar")
        return None
    
    # Convertir a DataFrame
    records = []
    request_numbers = []
    raw_ids = []
    
    for row_id, req_num, data_json in rows:
        records.append(data_json)
        request_numbers.append(req_num)
        raw_ids.append(row_id)
    
    df = pd.DataFrame(records)
    logger.info(f"📊 Obtenidos {len(df)} registros sin procesar")
    logger.info(f"   Peticiones: {set(request_numbers)}")
    
    context['task_instance'].xcom_push(key='raw_df', value=df.to_dict('records'))
    context['task_instance'].xcom_push(key='request_numbers', value=request_numbers)
    context['task_instance'].xcom_push(key='raw_ids', value=raw_ids)
    
    return len(df)

def clean_data(**context):
    """Limpieza de datos: outliers, tipos, valores nulos"""
    raw_data = context['task_instance'].xcom_pull(key='raw_df')
    df = pd.DataFrame(raw_data)
    
    initial_count = len(df)
    logger.info(f"🧹 Iniciando limpieza de {initial_count} registros")
    
    # 1. Filtrar outliers en price
    df = df[(df['price'] >= 50000) & (df['price'] <= 2000000)]
    logger.info(f"   Price outliers eliminados: {initial_count - len(df)}")
    
    # 2. Filtrar outliers en house_size
    df = df[(df['house_size'] >= 500) & (df['house_size'] <= 10000)]
    logger.info(f"   House_size outliers eliminados: {initial_count - len(df)}")
    
    # 3. Convertir tipos correctamente
    df['brokered_by'] = df['brokered_by'].astype(int).astype(str)
    df['zip_code'] = df['zip_code'].astype(int).astype(str)
    df['bed'] = df['bed'].astype(int)
    
    # 4. Eliminar columna street (alta cardinalidad, no útil)
    if 'street' in df.columns:
        df = df.drop('street', axis=1)
        logger.info("   Columna 'street' eliminada")
    
    # 5. Encoding de status (binario)
    le = LabelEncoder()
    df['status_encoded'] = le.fit_transform(df['status'])
    logger.info(f"   Status encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")
    
    final_count = len(df)
    logger.info(f"✅ Limpieza completada: {initial_count} → {final_count} registros")
    logger.info(f"   Eliminados: {initial_count - final_count} ({(initial_count - final_count)/initial_count*100:.1f}%)")
    
    context['task_instance'].xcom_push(key='clean_df', value=df.to_dict('records'))

def encode_features(**context):
    """Encoding de features categóricas"""
    clean_data = context['task_instance'].xcom_pull(key='clean_df')
    df = pd.DataFrame(clean_data)
    
    logger.info(f"🔢 Encoding de features categóricas")
    
    # Target Encoding para alta cardinalidad (city, brokered_by, zip_code)
    # Nota: En producción se usaría category_encoders, aquí simplificamos con mean encoding
    for col in ['city', 'brokered_by', 'zip_code']:
        if col in df.columns:
            target_mean = df.groupby(col)['price'].mean()
            df[f'{col}_encoded'] = df[col].map(target_mean)
            logger.info(f"   {col}: Target encoding aplicado ({len(target_mean)} categorías)")
    
    # One-Hot Encoding para baja cardinalidad (state)
    if 'state' in df.columns:
        state_dummies = pd.get_dummies(df['state'], prefix='state')
        df = pd.concat([df, state_dummies], axis=1)
        logger.info(f"   state: One-Hot encoding aplicado ({len(state_dummies.columns)} columnas)")
    
    initial_cols = context['task_instance'].xcom_pull(key='clean_df')
    initial_cols_count = len(pd.DataFrame(initial_cols).columns)
    final_cols = len(df.columns)
    
    logger.info(f"✅ Encoding completado: {initial_cols_count} → {final_cols} columnas")
    
    context['task_instance'].xcom_push(key='encoded_df', value=df.to_dict('records'))

def store_clean_data(**context):
    """Guardar datos procesados en CLEAN database"""
    encoded_data = context['task_instance'].xcom_pull(key='encoded_df')
    request_numbers = context['task_instance'].xcom_pull(key='request_numbers')
    
    df = pd.DataFrame(encoded_data)
    
    # Seleccionar solo columnas originales para CLEAN table
    base_columns = ['brokered_by', 'status', 'price', 'bed', 'bath', 'acre_lot', 
                    'city', 'state', 'zip_code', 'house_size']
    df_to_store = df[base_columns].copy()
    df_to_store['request_number'] = request_numbers[:len(df)]
    
    conn = get_clean_connection()
    cur = conn.cursor()
    
    # Preparar valores
    values = []
    for _, row in df_to_store.iterrows():
        values.append((
            row['request_number'],
            row['brokered_by'],
            row['status'],
            row['price'],
            row['bed'],
            row['bath'],
            row['acre_lot'],
            row['city'],
            row['state'],
            row['zip_code'],
            row['house_size']
        ))
    
    execute_values(
        cur,
        """
        INSERT INTO clean_data 
        (request_number, brokered_by, status, price, bed, bath, acre_lot, 
         city, state, zip_code, house_size)
        VALUES %s
        """,
        values
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info(f"💾 Almacenados {len(df_to_store)} registros en clean_data")

def mark_as_processed(**context):
    """Marcar registros RAW como procesados"""
    raw_ids = context['task_instance'].xcom_pull(key='raw_ids')
    
    conn = get_raw_connection()
    cur = conn.cursor()
    
    cur.execute("""
        UPDATE raw_data 
        SET used_in_training = TRUE 
        WHERE id = ANY(%s)
    """, (raw_ids,))
    
    conn.commit()
    cur.close()
    conn.close()
    
    logger.info(f"✅ Marcados {len(raw_ids)} registros como procesados")

with DAG(
    'data_processing',
    default_args=default_args,
    description='Procesamiento de datos RAW a CLEAN',
    schedule_interval=None,  # Trigger después de ingestion
    catchup=False,
    tags=['processing', 'grupo9']
) as dag:
    
    init_clean_db = PythonOperator(
        task_id='init_clean_database',
        python_callable=init_clean_database,
    )
    
    fetch_raw = PythonOperator(
        task_id='fetch_unprocessed_raw',
        python_callable=fetch_unprocessed_raw,
    )
    
    clean = PythonOperator(
        task_id='clean_data',
        python_callable=clean_data,
    )
    
    encode = PythonOperator(
        task_id='encode_features',
        python_callable=encode_features,
    )
    
    store_clean = PythonOperator(
        task_id='store_clean_data',
        python_callable=store_clean_data,
    )
    
    mark_processed = PythonOperator(
        task_id='mark_as_processed',
        python_callable=mark_as_processed,
    )
    
    init_clean_db >> fetch_raw >> clean >> encode >> store_clean >> mark_processed
