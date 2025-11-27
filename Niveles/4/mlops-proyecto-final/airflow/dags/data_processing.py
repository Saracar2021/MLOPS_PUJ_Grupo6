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
    """Initialize clean data database"""
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clean_data (
            id SERIAL PRIMARY KEY,
            request_number INTEGER,
            brokered_by TEXT,
            status TEXT,
            price FLOAT,
            bed INTEGER,
            bath FLOAT,
            acre_lot FLOAT,
            city TEXT,
            state TEXT,
            zip_code TEXT,
            house_size FLOAT,
            prev_sold_date TEXT,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            encoded BOOLEAN DEFAULT FALSE
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()
    print("Clean database initialized")

def fetch_unprocessed_raw(**context):
    """Fetch raw data that hasn't been processed yet"""
    
    # Connect to clean_data to get processed request_numbers
    conn_clean = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    cur_clean = conn_clean.cursor()
    
    # Get already processed request_numbers
    cur_clean.execute("SELECT DISTINCT request_number FROM clean_data")
    processed_requests = set(row[0] for row in cur_clean.fetchall())
    cur_clean.close()
    conn_clean.close()
    
    print(f"📊 Request numbers ya procesados: {processed_requests}")
    
    # Connect to raw_data
    conn_raw = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_RAW_DB', 'raw_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    cur_raw = conn_raw.cursor()
    
    # Get available request_numbers
    cur_raw.execute("SELECT DISTINCT request_number FROM raw_data ORDER BY request_number")
    all_requests = [row[0] for row in cur_raw.fetchall()]
    
    print(f"📊 Request numbers disponibles: {all_requests}")
    
    # Find unprocessed requests
    unprocessed_requests = [r for r in all_requests if r not in processed_requests]
    
    if not unprocessed_requests:
        print("⚠️ No hay datos nuevos para procesar")
        cur_raw.close()
        conn_raw.close()
        return 0
    
    # Get next unprocessed request
    next_request = unprocessed_requests[0]
    print(f"✅ Procesando request_number: {next_request}")
    
    # Fetch data for that request
    cur_raw.execute("""
        SELECT id, data, request_number
        FROM raw_data
        WHERE request_number = %s
    """, (next_request,))
    
    rows = cur_raw.fetchall()
    cur_raw.close()
    conn_raw.close()
    
    total_records = len(rows)
    print(f"📊 Obtenidos {total_records} registros sin procesar")
    print(f"   Peticiones: {{{next_request}}}")
    
    # Push to XCom
    context['task_instance'].xcom_push(key='raw_data', value=rows)
    context['task_instance'].xcom_push(key='request_number', value=next_request)
    
    return total_records

def clean_data(**context):
    """Clean and filter raw data"""
    raw_data_list = context['task_instance'].xcom_pull(task_ids='fetch_unprocessed_raw', key='raw_data')
    request_number = context['task_instance'].xcom_pull(task_ids='fetch_unprocessed_raw', key='request_number')
    
    if not raw_data_list:
        print("No data to clean")
        return
    
    # Convert to DataFrame - handle both string and dict
    records = []
    for record in raw_data_list:
        data = record[1]
        # If already a dict, use it; if string, parse it
        if isinstance(data, dict):
            records.append(data)
        else:
            records.append(json.loads(data))
    
    df = pd.DataFrame(records)
    original_count = len(df)
    print(f"📊 Datos originales: {original_count} registros")
    
    # Remove outliers
    df = df[
        (df['price'] >= 50000) & (df['price'] <= 2000000) &
        (df['house_size'] >= 500) & (df['house_size'] <= 10000)
    ]
    
    # Convert types
    df['brokered_by'] = df['brokered_by'].astype(str)
    df['zip_code'] = df['zip_code'].astype(str)
    
    # Drop street column
    if 'street' in df.columns:
        df = df.drop(columns=['street'])
    
    cleaned_count = len(df)
    removed = original_count - cleaned_count
    removed_pct = (removed / original_count * 100) if original_count > 0 else 0
    
    print(f"✅ Limpieza completada: {original_count} → {cleaned_count} registros")
    print(f"   Eliminados: {removed} ({removed_pct:.1f}%)")
    
    # Push cleaned data to XCom
    context['task_instance'].xcom_push(key='cleaned_df', value=df.to_json(orient='records'))
    context['task_instance'].xcom_push(key='request_number', value=request_number)
    
    return cleaned_count

def encode_features(**context):
    """Mark data as ready for encoding (actual encoding happens during training)"""
    request_number = context['task_instance'].xcom_pull(task_ids='clean_data', key='request_number')

    if request_number is None:
        print("⚠️ No request_number found, skipping")
        return

    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    cur = conn.cursor()

    cur.execute("""
        UPDATE clean_data
        SET encoded = TRUE
        WHERE request_number = %s
    """, (request_number,))

    rows_updated = cur.rowcount
    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Marked {rows_updated} records as encoded for request {request_number}")

def store_clean_data(**context):
    """Store cleaned data in database"""
    df_json = context['task_instance'].xcom_pull(task_ids='clean_data', key='cleaned_df')
    request_number = context['task_instance'].xcom_pull(task_ids='clean_data', key='request_number')

    if df_json is None:
        print("No data to store")
        return

    df = pd.read_json(df_json, orient='records')

    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    cur = conn.cursor()

    # Prepare data for insertion
    from psycopg2.extras import execute_values
    records = []
    for _, row in df.iterrows():
        records.append((
            request_number,
            str(row.get('brokered_by', '')),
            str(row.get('status', '')),
            float(row.get('price', 0)),
            int(row.get('bed', 0)),
            float(row.get('bath', 0)),
            float(row.get('acre_lot', 0)),
            str(row.get('city', '')),
            str(row.get('state', '')),
            str(row.get('zip_code', '')),
            float(row.get('house_size', 0)),
            str(row.get('prev_sold_date', ''))
        ))

    # Insert data
    execute_values(
        cur,
        """INSERT INTO clean_data
           (request_number, brokered_by, status, price, bed, bath, acre_lot,
            city, state, zip_code, house_size, prev_sold_date)
           VALUES %s""",
        records
    )

    conn.commit()
    cur.close()
    conn.close()

    print(f"✅ Stored {len(records)} records for request {request_number}")

def mark_as_processed(**context):
    """Mark raw data as processed (optional - we use request_number logic)"""
    request_number = context['task_instance'].xcom_pull(task_ids='store_clean_data', key='request_number')
    
    if request_number is None:
        request_number = context['task_instance'].xcom_pull(task_ids='fetch_unprocessed_raw', key='request_number')
    
    print(f"✅ Request {request_number} procesado exitosamente")
    print("   (Tracking basado en request_number, no en flags)")

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
