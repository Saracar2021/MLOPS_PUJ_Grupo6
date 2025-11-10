from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import json
import os
import re

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

MEDICATION_COLUMNS = [
    'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide', 'glimepiride',
    'acetohexamide', 'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
    'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
    'examide', 'citoglipton', 'insulin', 'glyburide-metformin', 'glipizide-metformin',
    'glimepiride-pioglitazone', 'metformin-rosiglitazone', 'metformin-pioglitazone'
]

def get_raw_conn():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_RAW_HOST'),
        port=os.getenv('POSTGRES_RAW_PORT'),
        database=os.getenv('POSTGRES_RAW_DB'),
        user=os.getenv('POSTGRES_RAW_USER'),
        password=os.getenv('POSTGRES_RAW_PASSWORD')
    )

def get_clean_conn():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_CLEAN_HOST'),
        port=os.getenv('POSTGRES_CLEAN_PORT'),
        database=os.getenv('POSTGRES_CLEAN_DB'),
        user=os.getenv('POSTGRES_CLEAN_USER'),
        password=os.getenv('POSTGRES_CLEAN_PASSWORD')
    )

def init_clean_database(**context):
    conn = get_clean_conn()
    cur = conn.cursor()
    
    cur.execute("""
        CREATE TABLE IF NOT EXISTS clean_data (
            id SERIAL PRIMARY KEY,
            batch_id INTEGER,
            split_type VARCHAR(10),
            record_id VARCHAR(50),
            features JSONB,
            target INTEGER,
            processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    conn.commit()
    cur.close()
    conn.close()

def age_to_numeric(age_str):
    match = re.search(r'\[(\d+)-(\d+)\)', age_str)
    if match:
        low = int(match.group(1))
        high = int(match.group(2))
        return (low + high) / 2
    return 55

def group_icd9(code):
    if pd.isna(code):
        return 'unknown'
    code_str = str(code)
    if code_str.startswith('250'):
        return 'diabetes'
    elif code_str.startswith(('390', '391', '392', '393', '394', '395', '396', '397', '398', '401', '402', '403', '404', '405', '410', '411', '412', '413', '414', '415', '416', '417', '420', '421', '422', '423', '424', '425', '426', '427', '428', '429', '430', '431', '432', '433', '434', '435', '436', '437', '438', '440', '441', '442', '443', '444', '445', '446', '447', '448', '449', '451', '452', '453', '454', '455', '456', '457', '458', '459')):
        return 'circulatory'
    elif code_str.startswith(('460', '461', '462', '463', '464', '465', '466', '470', '471', '472', '473', '474', '475', '476', '477', '478', '480', '481', '482', '483', '484', '485', '486', '487', '488', '490', '491', '492', '493', '494', '495', '496', '500', '501', '502', '503', '504', '505', '506', '507', '508', '510', '511', '512', '513', '514', '515', '516', '517', '518', '519')):
        return 'respiratory'
    elif code_str.startswith(('520', '521', '522', '523', '524', '525', '526', '527', '528', '529', '530', '531', '532', '533', '534', '535', '536', '537', '538', '540', '541', '542', '543', '550', '551', '552', '553', '555', '556', '557', '558', '560', '562', '564', '565', '566', '567', '568', '569', '570', '571', '572', '573', '574', '575', '576', '577', '578', '579')):
        return 'digestive'
    elif code_str.startswith(('580', '581', '582', '583', '584', '585', '586', '587', '588', '589', '590', '591', '592', '593', '594', '595', '596', '597', '598', '599', '600', '601', '602', '603', '604', '605', '606', '607', '608', '610', '611', '612', '614', '615', '616', '617', '618', '619', '620', '621', '622', '623', '624', '625', '626', '627', '628', '629', '630', '631', '632', '633', '634', '635', '636', '637', '638', '639')):
        return 'genitourinary'
    elif code_str.startswith(('710', '711', '712', '713', '714', '715', '716', '717', '718', '719', '720', '721', '722', '723', '724', '725', '726', '727', '728', '729', '730', '731', '732', '733', '734', '735', '736', '737', '738', '739')):
        return 'musculoskeletal'
    elif code_str.startswith(('800', '801', '802', '803', '804', '805', '806', '807', '808', '809', '810', '811', '812', '813', '814', '815', '816', '817', '818', '819', '820', '821', '822', '823', '824', '825', '826', '827', '828', '829', '830', '831', '832', '833', '834', '835', '836', '837', '838', '839', '840', '841', '842', '843', '844', '845', '846', '847', '848', '849', '850', '851', '852', '853', '854', '860', '861', '862', '863', '864', '865', '866', '867', '868', '869', '870', '871', '872', '873', '874', '875', '876', '877', '878', '879', '880', '881', '882', '883', '884', '885', '886', '887', '888', '889', '890', '891', '892', '893', '894', '895', '896', '897', '898', '899', '900', '901', '902', '903', '904', '905', '906', '907', '908', '909', '910', '911', '912', '913', '914', '915', '916', '917', '918', '919', '920', '921', '922', '923', '924', '925', '926', '927', '928', '929', '930', '931', '932', '933', '934', '935', '936', '937', '938', '939', '940', '941', '942', '943', '944', '945', '946', '947', '948', '949', '950', '951', '952', '953', '954', '955', '956', '957', '958', '959')):
        return 'injury'
    else:
        return 'other'

def preprocess_record(record):
    data = json.loads(record['data'])
    df = pd.Series(data)
    
    if df.get('gender') in ['Unknown/Invalid']:
        return None
    
    features = {}
    
    features['age_numeric'] = age_to_numeric(df.get('age', '[50-60)'))
    
    for col in ['race', 'gender']:
        features[col] = df.get(col, 'Unknown')
    
    if pd.isna(features['race']) or features['race'] == '?':
        features['race'] = 'Other/Unknown'
    
    for col in ['payer_code', 'medical_specialty']:
        val = df.get(col)
        features[col] = val if pd.notna(val) and val != '?' else 'Unknown'
    
    for col in ['diag_1', 'diag_2', 'diag_3']:
        features[f'{col}_grouped'] = group_icd9(df.get(col))
    
    max_glu = df.get('max_glu_serum', 'None')
    glu_map = {'None': 0, 'Norm': 1, '>200': 2, '>300': 3}
    features['max_glu_serum'] = glu_map.get(max_glu, 0)
    
    a1c = df.get('A1Cresult', 'None')
    a1c_map = {'None': 0, 'Norm': 1, '>7': 2, '>8': 3}
    features['A1Cresult'] = a1c_map.get(a1c, 0)
    
    med_map = {'No': 0, 'Steady': 1, 'Down': 2, 'Up': 3}
    for med in MEDICATION_COLUMNS:
        val = df.get(med, 'No')
        features[med] = med_map.get(val, 0)
    
    features['change'] = 1 if df.get('change') in ['Ch', 'Yes'] else 0
    features['diabetesMed'] = 1 if df.get('diabetesMed') == 'Yes' else 0
    
    for col in ['admission_type_id', 'discharge_disposition_id', 'admission_source_id',
                'time_in_hospital', 'num_lab_procedures', 'num_procedures', 'num_medications',
                'number_outpatient', 'number_emergency', 'number_inpatient', 'number_diagnoses']:
        features[col] = int(df.get(col, 0))
    
    readmitted = df.get('readmitted', 'NO')
    target_map = {'NO': 0, '>30': 1, '<30': 2}
    target = target_map.get(readmitted, 0)
    
    return features, target

def process_batch(**context):
    raw_conn = get_raw_conn()
    clean_conn = get_clean_conn()
    
    raw_cur = raw_conn.cursor()
    clean_cur = clean_conn.cursor()
    
    clean_cur.execute("SELECT COALESCE(MAX(batch_id), 0) FROM clean_data")
    last_processed = clean_cur.fetchone()[0]
    
    raw_cur.execute("""
        SELECT DISTINCT batch_id FROM raw_data 
        WHERE batch_id > %s 
        ORDER BY batch_id 
        LIMIT 1
    """, (last_processed,))
    
    batch_result = raw_cur.fetchone()
    if not batch_result:
        print("No new batches to process")
        raw_cur.close()
        clean_cur.close()
        raw_conn.close()
        clean_conn.close()
        return
    
    current_batch = batch_result[0]
    
    raw_cur.execute("""
        SELECT batch_id, split_type, record_id, data 
        FROM raw_data 
        WHERE batch_id = %s
    """, (current_batch,))
    
    records = raw_cur.fetchall()
    processed_records = []
    
    for record in records:
        result = preprocess_record({
            'batch_id': record[0],
            'split_type': record[1],
            'record_id': record[2],
            'data': record[3]
        })
        
        if result:
            features, target = result
            processed_records.append((
                record[0],
                record[1],
                record[2],
                json.dumps(features),
                target
            ))
    
    if processed_records:
        execute_values(
            clean_cur,
            "INSERT INTO clean_data (batch_id, split_type, record_id, features, target) VALUES %s",
            processed_records
        )
        clean_conn.commit()
        print(f"Processed batch {current_batch}: {len(processed_records)} records")
    
    raw_cur.close()
    clean_cur.close()
    raw_conn.close()
    clean_conn.close()

with DAG(
    'data_processing_pipeline',
    default_args=default_args,
    description='Process RAW data to CLEAN data',
    schedule_interval='*/15 * * * *',
    catchup=False,
) as dag:
    
    init_clean_db = PythonOperator(
        task_id='init_clean_database',
        python_callable=init_clean_database,
    )
    
    process = PythonOperator(
        task_id='process_batch',
        python_callable=process_batch,
    )
    
    init_clean_db >> process
