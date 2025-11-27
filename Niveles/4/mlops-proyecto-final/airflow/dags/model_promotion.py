from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import mlflow
from mlflow.tracking import MlflowClient
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

MLFLOW_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
EXPERIMENT_NAME = "real_estate_prediction"
MODEL_NAME = "real_estate_model"

def promote_best_model(**context):
    """Encontrar y promocionar mejor modelo a Production"""
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()
    
    # Obtener experimento
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        logger.error(f"❌ Experimento '{EXPERIMENT_NAME}' no encontrado")
        raise ValueError(f"Experiment {EXPERIMENT_NAME} not found")
    
    # Buscar mejor run por RMSE
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.rmse ASC"],
        max_results=1
    )
    
    if not runs:
        logger.warning("⚠️ No hay runs disponibles")
        return
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_type = best_run.data.params.get('model_type', 'Unknown')
    rmse = best_run.data.metrics['rmse']
    r2 = best_run.data.metrics.get('r2_score', 0)
    
    logger.info(f"📊 Mejor modelo encontrado:")
    logger.info(f"   Tipo: {model_type}")
    logger.info(f"   RMSE: {rmse:.2f}")
    logger.info(f"   R² Score: {r2:.3f}")
    logger.info(f"   Run ID: {run_id}")
    
    # Registrar modelo si no existe
    try:
        client.get_registered_model(MODEL_NAME)
        logger.info(f"   Modelo '{MODEL_NAME}' ya existe")
    except:
        client.create_registered_model(MODEL_NAME)
        logger.info(f"   Modelo '{MODEL_NAME}' creado")
    
    # Crear nueva versión
    model_uri = f"runs:/{run_id}/model"
    model_version = client.create_model_version(
        name=MODEL_NAME,
        source=model_uri,
        run_id=run_id
    )
    
    logger.info(f"✅ Versión {model_version.version} registrada")
    
    # Promocionar a Production
    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    logger.info(f"🚀 Modelo {model_type} v{model_version.version} promovido a PRODUCTION")
    logger.info(f"   RMSE: {rmse:.2f} | R²: {r2:.3f}")
    
    # Guardar en XCom
    context['task_instance'].xcom_push(key='model_version', value=model_version.version)
    context['task_instance'].xcom_push(key='model_type', value=model_type)
    context['task_instance'].xcom_push(key='rmse', value=rmse)

with DAG(
    'model_promotion',
    default_args=default_args,
    description='Promoción del mejor modelo a Production',
    schedule_interval=None,
    catchup=False,
    tags=['promotion', 'grupo9']
) as dag:
    
    promote = PythonOperator(
        task_id='promote_best_model',
        python_callable=promote_best_model,
    )
