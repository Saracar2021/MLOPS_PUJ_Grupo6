from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import os

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

BATCH_SIZE = 15000

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_RAW_HOST'),
        port=os.getenv('POSTGRES_RAW_PORT'),
        database=os.getenv('POSTGRES_RAW_DB'),
        user=os.getenv('POSTGRES_RAW_USER'),
        password=os.getenv('POSTGRES_RAW_PASSWORD')
    )

def init_database(**context):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS raw_data (
            id SERIAL PRIMARY KEY,
            batch_id INTEGER,
            split_type VARCHAR(10),
            record_id VARCHAR(50),
            data JSONB,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS batch_control (
            batch_id INTEGER PRIMARY KEY,
            total_records INTEGER,
            loaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def load_and_split_batch(**context):
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute("SELECT COALESCE(MAX(batch_id), 0) FROM batch_control")
    last_batch = cur.fetchone()[0]
    current_batch = last_batch + 1
    
    if current_batch > 7:
        print("All batches loaded")
        return
    
    diabetes = fetch_ucirepo(id=296)
    X = diabetes.data.features
    y = diabetes.data.targets
    
    df = pd.concat([X, y], axis=1)
    
    start_idx = (current_batch - 1) * BATCH_SIZE
    end_idx = current_batch * BATCH_SIZE
    batch_df = df.iloc[start_idx:end_idx].copy()
    
    batch_df['record_id'] = [f"rec_{start_idx + i}" for i in range(len(batch_df))]
    
    train_df, temp_df = train_test_split(
        batch_df, 
        test_size=0.3, 
        stratify=batch_df['readmitted'],
        random_state=42 + current_batch
    )
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.5,
        stratify=temp_df['readmitted'],
        random_state=42 + current_batch
    )
    
    records = []
    for split_type, split_df in [('train', train_df), ('val', val_df), ('test', test_df)]:
        for _, row in split_df.iterrows():
            records.append((
                current_batch,
                split_type,
                row['record_id'],
                row.drop('record_id').to_json()
            ))
    
    execute_values(
        cur,
        "INSERT INTO raw_data (batch_id, split_type, record_id, data) VALUES %s",
        records
    )
    
    cur.execute(
        "INSERT INTO batch_control (batch_id, total_records) VALUES (%s, %s)",
        (current_batch, len(batch_df))
    )
    
    conn.commit()
    cur.close()
    conn.close()
    
    print(f"Batch {current_batch} loaded: {len(train_df)} train, {len(val_df)} val, {len(test_df)} test")

with DAG(
    'data_ingestion_batch',
    default_args=default_args,
    description='Load diabetes dataset in 15k batches',
    schedule_interval='*/10 * * * *',
    catchup=False,
) as dag:
    
    init_db = PythonOperator(
        task_id='init_database',
        python_callable=init_database,
    )
    
    load_batch = PythonOperator(
        task_id='load_and_split_batch',
        python_callable=load_and_split_batch,
    )
    
    init_db >> load_batch
