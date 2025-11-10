from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import os
import mlflow
from mlflow.tracking import MlflowClient

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def promote_best_model(**context):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    client = MlflowClient()
    
    experiment = client.get_experiment_by_name("diabetes_readmission")
    if not experiment:
        print("Experiment not found")
        return
    
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1_score_weighted DESC"],
        max_results=1
    )
    
    if not runs:
        print("No runs found")
        return
    
    best_run = runs[0]
    run_id = best_run.info.run_id
    model_uri = f"runs:/{run_id}/model"
    
    model_name = "diabetes_readmission_model"
    
    try:
        registered_model = client.get_registered_model(model_name)
    except:
        registered_model = client.create_registered_model(model_name)
    
    model_version = client.create_model_version(
        name=model_name,
        source=model_uri,
        run_id=run_id
    )
    
    client.transition_model_version_stage(
        name=model_name,
        version=model_version.version,
        stage="Production",
        archive_existing_versions=True
    )
    
    print(f"Model version {model_version.version} promoted to Production")
    print(f"F1-score: {best_run.data.metrics['f1_score_weighted']}")
    print(f"Model type: {best_run.data.params.get('model_type', 'Unknown')}")

with DAG(
    'model_promotion_pipeline',
    default_args=default_args,
    description='Promote best model to Production',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    promote = PythonOperator(
        task_id='promote_best_model',
        python_callable=promote_best_model,
    )
