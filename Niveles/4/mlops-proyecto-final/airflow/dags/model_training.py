from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
import pandas as pd
import psycopg2
import os
import logging
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ks_2samp
import numpy as np

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

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = "real_estate_prediction"

def get_clean_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database='clean_data',
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )

def setup_mlflow():
    """Configurar MLflow"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    try:
        mlflow.create_experiment(EXPERIMENT_NAME)
    except:
        pass
    mlflow.set_experiment(EXPERIMENT_NAME)
    logger.info(f"✅ MLflow configurado: {MLFLOW_URI}")

def should_retrain(**context):
    """Decidir si entrenar basado en drift y cantidad"""
    conn = get_clean_connection()
    
    # Obtener datos baseline (primera petición)
    df_baseline = pd.read_sql_query(
        "SELECT * FROM clean_data WHERE request_number = 1",
        conn
    )
    
    # Obtener datos nuevos
    df_new = pd.read_sql_query(
        """SELECT * FROM clean_data 
           WHERE request_number = (SELECT MAX(request_number) FROM clean_data)""",
        conn
    )
    
    conn.close()
    
    # Criterio 1: Cantidad mínima
    if len(df_new) < 10000:
        reason = f"Insuficientes datos ({len(df_new)} < 10,000)"
        logger.info(f"⏭️ SKIP ENTRENAMIENTO: {reason}")
        context['task_instance'].xcom_push(key='retrain_reason', value=reason)
        return 'end_without_training'
    
    # Criterio 2: Drift detection
    numeric_features = ['price', 'house_size', 'bed', 'bath']
    drift_detected = False
    drift_feature = None
    drift_pvalue = None
    
    for feature in numeric_features:
        stat, p_value = ks_2samp(df_baseline[feature], df_new[feature])
        
        if p_value < 0.05:
            drift_detected = True
            drift_feature = feature
            drift_pvalue = p_value
            break
    
    if drift_detected:
        reason = f"DRIFT en '{drift_feature}' (p-value={drift_pvalue:.4f})"
        logger.info(f"✅ REENTRENAMIENTO JUSTIFICADO: {reason}")
        context['task_instance'].xcom_push(key='retrain_reason', value=reason)
        return 'load_training_data'
    else:
        reason = "No drift significativo detectado"
        logger.info(f"⏭️ SKIP ENTRENAMIENTO: {reason}")
        context['task_instance'].xcom_push(key='retrain_reason', value=reason)
        return 'end_without_training'

def load_training_data(**context):
    """Cargar todos los datos para entrenamiento"""
    conn = get_clean_connection()
    df = pd.read_sql_query("SELECT * FROM clean_data", conn)
    conn.close()
    
    logger.info(f"📊 Datos cargados: {len(df)} registros")
    
    # Separar features y target
    target = 'price'
    features = [col for col in df.columns if col not in ['id', 'request_number', 'processed_at', target]]
    
    X = df[features]
    y = df[target]
    
    # Split train/val/test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    logger.info(f"   Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    context['task_instance'].xcom_push(key='X_train', value=X_train.to_dict('records'))
    context['task_instance'].xcom_push(key='y_train', value=y_train.tolist())
    context['task_instance'].xcom_push(key='X_val', value=X_val.to_dict('records'))
    context['task_instance'].xcom_push(key='y_val', value=y_val.tolist())
    context['task_instance'].xcom_push(key='feature_names', value=features)

def train_models(**context):
    """Entrenar 3 modelos y registrar en MLflow"""
    setup_mlflow()
    
    # Cargar datos
    X_train = pd.DataFrame(context['task_instance'].xcom_pull(key='X_train'))
    y_train = np.array(context['task_instance'].xcom_pull(key='y_train'))
    X_val = pd.DataFrame(context['task_instance'].xcom_pull(key='X_val'))
    y_val = np.array(context['task_instance'].xcom_pull(key='y_val'))
    
    # Modelos a entrenar
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    best_model_name = None
    best_rmse = float('inf')
    
    for name, model in models.items():
        logger.info(f"🤖 Entrenando {name}...")
        
        with mlflow.start_run(run_name=name):
            # Entrenar
            model.fit(X_train, y_train)
            
            # Predecir
            y_pred = model.predict(X_val)
            
            # Métricas
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            mape = np.mean(np.abs((y_val - y_pred) / y_val)) * 100
            
            # Log en MLflow
            mlflow.log_param("model_type", name)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2_score", r2)
            mlflow.log_metric("mape", mape)
            
            # Registrar modelo
            if name == 'XGBoost':
                mlflow.xgboost.log_model(model, "model")
            else:
                mlflow.sklearn.log_model(model, "model")
            
            logger.info(f"   {name}: RMSE={rmse:.2f} | MAE={mae:.2f} | R²={r2:.3f}")
            
            # Trackear mejor modelo
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = name
    
    logger.info(f"✅ Mejor modelo: {best_model_name} (RMSE={best_rmse:.2f})")
    context['task_instance'].xcom_push(key='best_model', value=best_model_name)
    context['task_instance'].xcom_push(key='best_rmse', value=best_rmse)

def end_without_training(**context):
    """Finalizar sin entrenar"""
    reason = context['task_instance'].xcom_pull(key='retrain_reason')
    logger.info(f"⏭️ Pipeline finalizado sin entrenamiento: {reason}")

with DAG(
    'model_training',
    default_args=default_args,
    description='Entrenamiento de modelos con drift detection',
    schedule_interval=None,
    catchup=False,
    tags=['training', 'grupo9']
) as dag:
    
    decide = BranchPythonOperator(
        task_id='should_retrain',
        python_callable=should_retrain,
    )
    
    load_data = PythonOperator(
        task_id='load_training_data',
        python_callable=load_training_data,
    )
    
    train = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
    )
    
    end_skip = PythonOperator(
        task_id='end_without_training',
        python_callable=end_without_training,
    )
    
    decide >> [load_data, end_skip]
    load_data >> train
