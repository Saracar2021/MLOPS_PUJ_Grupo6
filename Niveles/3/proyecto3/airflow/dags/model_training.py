from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
import pandas as pd
import json
import os
import sys
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')  # Set backend before importing pyplot
import matplotlib.pyplot as plt
import seaborn as sns
import io
import gc
import logging
import traceback

# Configure logging - force stdout with immediate flush
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Ensure stdout is unbuffered
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Log that DAG is being loaded
print("=" * 80, flush=True)
print("Loading model_training_pipeline DAG...", flush=True)
print("=" * 80, flush=True)

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(minutes=10),
}

def get_clean_conn():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_CLEAN_HOST'),
        port=os.getenv('POSTGRES_CLEAN_PORT'),
        database=os.getenv('POSTGRES_CLEAN_DB'),
        user=os.getenv('POSTGRES_CLEAN_USER'),
        password=os.getenv('POSTGRES_CLEAN_PASSWORD')
    )

def setup_mlflow():
    """Setup MLflow tracking URI and experiment with proper error handling"""
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI')
    logger.info(f"MLflow URI: {mlflow_uri}")

    if not mlflow_uri:
        raise ValueError("MLFLOW_TRACKING_URI environment variable is not set")

    mlflow.set_tracking_uri(mlflow_uri)

    # Get or create experiment (handles concurrent creation)
    experiment_name = "diabetes_readmission"
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            logger.info(f"Creating new experiment: {experiment_name}")
            mlflow.create_experiment(experiment_name)
        else:
            logger.info(f"Using existing experiment: {experiment_name}")
    except Exception as e:
        logger.warning(f"Error creating/getting experiment: {e}. Will use set_experiment as fallback")

    mlflow.set_experiment(experiment_name)

def load_data(split_type):
    conn = get_clean_conn()
    cur = conn.cursor()
    
    cur.execute("""
        SELECT features, target 
        FROM clean_data 
        WHERE split_type = %s
    """, (split_type,))
    
    rows = cur.fetchall()
    cur.close()
    conn.close()
    
    features_list = []
    targets = []

    for row in rows:
        # PostgreSQL may return JSON as dict or string depending on configuration
        if isinstance(row[0], dict):
            features = row[0]
        else:
            features = json.loads(row[0])
        features_list.append(features)
        targets.append(row[1])

    return pd.DataFrame(features_list), pd.Series(targets)

def prepare_pipeline(X):
    categorical_features = ['race', 'gender', 'payer_code', 'medical_specialty',
                           'diag_1_grouped', 'diag_2_grouped', 'diag_3_grouped']
    
    numeric_features = ['age_numeric', 'admission_type_id', 'discharge_disposition_id',
                       'admission_source_id', 'time_in_hospital', 'num_lab_procedures',
                       'num_procedures', 'num_medications', 'number_outpatient',
                       'number_emergency', 'number_inpatient', 'number_diagnoses',
                       'max_glu_serum', 'A1Cresult', 'change', 'diabetesMed']
    
    medication_features = [col for col in X.columns if col not in categorical_features + numeric_features]
    numeric_features.extend(medication_features)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    return preprocessor

def train_logistic_regression(**context):
    try:
        print("=" * 80, flush=True)
        print("TASK STARTED: LogisticRegression Training", flush=True)
        print("=" * 80, flush=True)
        logger.info("Starting LogisticRegression training")

        # Setup MLflow
        setup_mlflow()

        logger.info("Loading training data")
        X_train, y_train = load_data('train')
        logger.info(f"Training data loaded: {X_train.shape[0]} samples")

        logger.info("Loading validation data")
        X_val, y_val = load_data('val')
        logger.info(f"Validation data loaded: {X_val.shape[0]} samples")

        logger.info("Preparing preprocessing pipeline")
        preprocessor = prepare_pipeline(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        logger.info(f"Data preprocessed: {X_train_processed.shape[1]} features")

        with mlflow.start_run(run_name="LogisticRegression"):
            logger.info("Training LogisticRegression model")
            model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
            model.fit(X_train_processed, y_train)

            logger.info("Making predictions")
            y_pred = model.predict(X_val_processed)

            logger.info("Calculating metrics")
            f1 = f1_score(y_val, y_pred, average='weighted')
            accuracy = accuracy_score(y_val, y_pred)
            recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)

            logger.info(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Recall Class 2: {recall_class_2:.4f}")

            mlflow.log_param("model_type", "LogisticRegression")
            mlflow.log_param("max_iter", 1000)
            mlflow.log_metric("f1_score_weighted", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall_class_2", recall_class_2)

            logger.info("Creating confusion matrix")
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Confusion Matrix - LogisticRegression')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()

            logger.info("Logging model to MLflow")
            mlflow.sklearn.log_model(model, "model", input_example=X_train_processed[:1])

            run = mlflow.active_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run completed: {run_id}")

            context['task_instance'].xcom_push(key='lr_run_id', value=run_id)
            context['task_instance'].xcom_push(key='lr_f1_score', value=f1)

        # Clean up memory
        del X_train, y_train, X_val, y_val, X_train_processed, X_val_processed, model, preprocessor
        gc.collect()

        logger.info("LogisticRegression training completed successfully")
        print("=" * 80, flush=True)
        print("TASK COMPLETED: LogisticRegression Training - SUCCESS", flush=True)
        print(f"F1 Score: {f1:.4f}, Run ID: {run_id}", flush=True)
        print("=" * 80, flush=True)
        return {"status": "success", "f1_score": f1, "run_id": run_id}

    except Exception as e:
        print("=" * 80, flush=True)
        print("TASK FAILED: LogisticRegression Training", flush=True)
        print(f"Error: {str(e)}", flush=True)
        print("=" * 80, flush=True)
        logger.error(f"Error in train_logistic_regression: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_random_forest(**context):
    try:
        print("=" * 80, flush=True)
        print("TASK STARTED: RandomForest Training", flush=True)
        print("=" * 80, flush=True)
        logger.info("Starting RandomForest training")

        # Setup MLflow
        setup_mlflow()

        logger.info("Loading training data")
        X_train, y_train = load_data('train')
        logger.info(f"Training data loaded: {X_train.shape[0]} samples")

        logger.info("Loading validation data")
        X_val, y_val = load_data('val')
        logger.info(f"Validation data loaded: {X_val.shape[0]} samples")

        logger.info("Preparing preprocessing pipeline")
        preprocessor = prepare_pipeline(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        logger.info(f"Data preprocessed: {X_train_processed.shape[1]} features")

        with mlflow.start_run(run_name="RandomForest"):
            logger.info("Training RandomForest model")
            model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
            model.fit(X_train_processed, y_train)

            logger.info("Making predictions")
            y_pred = model.predict(X_val_processed)

            logger.info("Calculating metrics")
            f1 = f1_score(y_val, y_pred, average='weighted')
            accuracy = accuracy_score(y_val, y_pred)
            recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)

            logger.info(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Recall Class 2: {recall_class_2:.4f}")

            mlflow.log_param("model_type", "RandomForest")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 10)
            mlflow.log_metric("f1_score_weighted", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall_class_2", recall_class_2)

            logger.info("Creating confusion matrix")
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax)
            ax.set_title('Confusion Matrix - RandomForest')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()

            logger.info("Logging model to MLflow")
            mlflow.sklearn.log_model(model, "model", input_example=X_train_processed[:1])

            run = mlflow.active_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run completed: {run_id}")

            context['task_instance'].xcom_push(key='rf_run_id', value=run_id)
            context['task_instance'].xcom_push(key='rf_f1_score', value=f1)

        # Clean up memory
        del X_train, y_train, X_val, y_val, X_train_processed, X_val_processed, model, preprocessor
        gc.collect()

        logger.info("RandomForest training completed successfully")
        print("=" * 80, flush=True)
        print("TASK COMPLETED: RandomForest Training - SUCCESS", flush=True)
        print(f"F1 Score: {f1:.4f}, Run ID: {run_id}", flush=True)
        print("=" * 80, flush=True)
        return {"status": "success", "f1_score": f1, "run_id": run_id}

    except Exception as e:
        print("=" * 80, flush=True)
        print("TASK FAILED: RandomForest Training", flush=True)
        print(f"Error: {str(e)}", flush=True)
        print("=" * 80, flush=True)
        logger.error(f"Error in train_random_forest: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def train_xgboost(**context):
    try:
        print("=" * 80, flush=True)
        print("TASK STARTED: XGBoost Training", flush=True)
        print("=" * 80, flush=True)
        logger.info("Starting XGBoost training")

        # Setup MLflow
        setup_mlflow()

        logger.info("Loading training data")
        X_train, y_train = load_data('train')
        logger.info(f"Training data loaded: {X_train.shape[0]} samples")

        logger.info("Loading validation data")
        X_val, y_val = load_data('val')
        logger.info(f"Validation data loaded: {X_val.shape[0]} samples")

        logger.info("Preparing preprocessing pipeline")
        preprocessor = prepare_pipeline(X_train)
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        logger.info(f"Data preprocessed: {X_train_processed.shape[1]} features")

        with mlflow.start_run(run_name="XGBoost"):
            logger.info("Training XGBoost model")
            model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
            model.fit(X_train_processed, y_train)

            logger.info("Making predictions")
            y_pred = model.predict(X_val_processed)

            logger.info("Calculating metrics")
            f1 = f1_score(y_val, y_pred, average='weighted')
            accuracy = accuracy_score(y_val, y_pred)
            recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)

            logger.info(f"F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Recall Class 2: {recall_class_2:.4f}")

            mlflow.log_param("model_type", "XGBoost")
            mlflow.log_param("n_estimators", 100)
            mlflow.log_param("max_depth", 6)
            mlflow.log_param("learning_rate", 0.1)
            mlflow.log_metric("f1_score_weighted", f1)
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("recall_class_2", recall_class_2)

            logger.info("Creating confusion matrix")
            cm = confusion_matrix(y_val, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', ax=ax)
            ax.set_title('Confusion Matrix - XGBoost')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close()

            logger.info("Logging model to MLflow")
            mlflow.xgboost.log_model(model, "model", input_example=X_train_processed[:1])

            run = mlflow.active_run()
            run_id = run.info.run_id
            logger.info(f"MLflow run completed: {run_id}")

            context['task_instance'].xcom_push(key='xgb_run_id', value=run_id)
            context['task_instance'].xcom_push(key='xgb_f1_score', value=f1)

        # Clean up memory
        del X_train, y_train, X_val, y_val, X_train_processed, X_val_processed, model, preprocessor
        gc.collect()

        logger.info("XGBoost training completed successfully")
        print("=" * 80, flush=True)
        print("TASK COMPLETED: XGBoost Training - SUCCESS", flush=True)
        print(f"F1 Score: {f1:.4f}, Run ID: {run_id}", flush=True)
        print("=" * 80, flush=True)
        return {"status": "success", "f1_score": f1, "run_id": run_id}

    except Exception as e:
        print("=" * 80, flush=True)
        print("TASK FAILED: XGBoost Training", flush=True)
        print(f"Error: {str(e)}", flush=True)
        print("=" * 80, flush=True)
        logger.error(f"Error in train_xgboost: {str(e)}")
        logger.error(traceback.format_exc())
        raise

with DAG(
    'model_training_pipeline',
    default_args=default_args,
    description='Train models and register in MLflow',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    train_lr = PythonOperator(
        task_id='train_logistic_regression',
        python_callable=train_logistic_regression,
    )
    
    train_rf = PythonOperator(
        task_id='train_random_forest',
        python_callable=train_random_forest,
    )
    
    train_xgb = PythonOperator(
        task_id='train_xgboost',
        python_callable=train_xgboost,
    )

    [train_lr, train_rf, train_xgb]

# Log that DAG was loaded successfully
print("=" * 80, flush=True)
print("model_training_pipeline DAG loaded successfully!", flush=True)
print("Tasks: train_logistic_regression, train_random_forest, train_xgboost", flush=True)
print("=" * 80, flush=True)
