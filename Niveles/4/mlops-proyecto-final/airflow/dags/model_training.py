from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import psycopg2
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import LabelEncoder
from scipy import stats
import numpy as np

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

EXPERIMENT_NAME = "real_estate_prediction"
MODEL_NAME = "real_estate_model"
MIN_RECORDS_FOR_TRAINING = 10000
DRIFT_THRESHOLD = 0.05

def check_drift_and_decide(**context):
    """Check if retraining is needed based on drift detection"""
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    client = MlflowClient()
    
    # Check if this is first training (no experiment exists)
    try:
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        if experiment is None:
            print("🆕 PRIMERA EJECUCIÓN: Entrenar modelo baseline")
            return 'train_models'
        
        # Check if there are any successful runs
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="status = 'FINISHED'",
            max_results=1
        )
        
        if not runs:
            print("🆕 NO HAY MODELOS PREVIOS: Entrenar modelo baseline")
            return 'train_models'
            
    except Exception as e:
        print(f"🆕 EXPERIMENTO NO EXISTE: Entrenar modelo baseline - {str(e)}")
        return 'train_models'
    
    # If we get here, there are previous models - check drift
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    
    # Get all data
    df = pd.read_sql("SELECT * FROM clean_data WHERE encoded = TRUE", conn)
    conn.close()
    
    total_records = len(df)
    print(f"📊 Total records: {total_records}")
    
    if total_records < MIN_RECORDS_FOR_TRAINING:
        print(f"⏭️ SKIP ENTRENAMIENTO: Insuficientes datos ({total_records} < {MIN_RECORDS_FOR_TRAINING})")
        return 'end_without_training'
    
    # Check drift on key features
    drift_features = ['price', 'house_size', 'bed', 'bath']
    drift_detected = False
    
    # Split into old and new data (last 20% as new)
    split_idx = int(len(df) * 0.8)
    old_data = df.iloc[:split_idx]
    new_data = df.iloc[split_idx:]
    
    for feature in drift_features:
        if feature in df.columns:
            ks_stat, p_value = stats.ks_2samp(old_data[feature], new_data[feature])
            print(f"🔍 Drift test {feature}: KS={ks_stat:.4f}, p-value={p_value:.4f}")
            
            if p_value < DRIFT_THRESHOLD:
                print(f"⚠️ DRIFT DETECTADO en {feature}")
                drift_detected = True
    
    if drift_detected:
        print("✅ ENTRENAR: Drift detectado")
        return 'train_models'
    else:
        print("⏭️ SKIP: No drift detectado")
        return 'end_without_training'

def train_models(**context):
    """Train multiple regression models and log to MLflow"""
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000'))
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Load data
    conn = psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=os.getenv('POSTGRES_PORT', '5432'),
        database=os.getenv('POSTGRES_CLEAN_DB', 'clean_data'),
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )
    
    df = pd.read_sql("SELECT * FROM clean_data WHERE encoded = TRUE", conn)
    conn.close()
    
    print(f"📊 Loaded {len(df)} records")
    
    # Prepare features
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = ['brokered_by', 'status', 'city', 'state', 'zip_code']
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # Select features and target
    feature_cols = ['bed', 'bath', 'acre_lot', 'house_size', 'brokered_by', 'status', 'city', 'state', 'zip_code']
    X = df[feature_cols]
    y = df['price']
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    print(f"📊 Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    # Train models
    models = {
        'Ridge': Ridge(alpha=1.0),
        'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
        'XGBoost': XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    }
    
    best_model_name = None
    best_rmse = float('inf')
    
    for model_name, model in models.items():
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            print(f"🤖 Training {model_name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_val)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
            mape = mean_absolute_percentage_error(y_val, y_pred)
            
            print(f"📈 {model_name} - RMSE: {rmse:.2f}, MAE: {mae:.2f}, R²: {r2:.4f}, MAPE: {mape:.4f}")
            
            # Log to MLflow
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("n_train", len(X_train))
            mlflow.log_param("n_val", len(X_val))
            
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("r2", r2)
            mlflow.log_metric("mape", mape)
            
            # Log model
            mlflow.sklearn.log_model(model, "model")
            
            # Track best model
            if rmse < best_rmse:
                best_rmse = rmse
                best_model_name = model_name
    
    print(f"🏆 Best model: {best_model_name} with RMSE: {best_rmse:.2f}")
    context['task_instance'].xcom_push(key='best_model', value=best_model_name)
    context['task_instance'].xcom_push(key='best_rmse', value=best_rmse)

with DAG(
    'model_training',
    default_args=default_args,
    description='Train models with drift detection',
    schedule_interval='@daily',
    catchup=False,
) as dag:
    
    check_drift = BranchPythonOperator(
        task_id='check_drift_and_decide',
        python_callable=check_drift_and_decide,
    )
    
    train = PythonOperator(
        task_id='train_models',
        python_callable=train_models,
    )
    
    end_without = EmptyOperator(
        task_id='end_without_training'
    )
    
    end_with = EmptyOperator(
        task_id='end_with_training',
        trigger_rule='none_failed_min_one_success'
    )
    
    check_drift >> [train, end_without]
    train >> end_with
    end_without >> end_with
