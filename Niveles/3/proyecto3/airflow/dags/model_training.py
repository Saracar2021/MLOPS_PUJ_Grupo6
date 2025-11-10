from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import psycopg2
import pandas as pd
import json
import os
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import f1_score, accuracy_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import io

default_args = {
    'owner': 'mlops_grupo6',
    'depends_on_past': False,
    'start_date': datetime(2025, 11, 9),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def get_clean_conn():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_CLEAN_HOST'),
        port=os.getenv('POSTGRES_CLEAN_PORT'),
        database=os.getenv('POSTGRES_CLEAN_DB'),
        user=os.getenv('POSTGRES_CLEAN_USER'),
        password=os.getenv('POSTGRES_CLEAN_PASSWORD')
    )

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
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("diabetes_readmission")
    
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    
    preprocessor = prepare_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    with mlflow.start_run(run_name="LogisticRegression"):
        model = LogisticRegression(max_iter=1000, random_state=42, multi_class='multinomial')
        model.fit(X_train_processed, y_train)
        
        y_pred = model.predict(X_val_processed)
        
        f1 = f1_score(y_val, y_pred, average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)
        
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 1000)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall_class_2", recall_class_2)
        
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
        
        mlflow.sklearn.log_model(model, "model", input_example=X_train_processed[:5])
        
        run = mlflow.active_run()
        context['task_instance'].xcom_push(key='lr_run_id', value=run.info.run_id)
        context['task_instance'].xcom_push(key='lr_f1_score', value=f1)

def train_random_forest(**context):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("diabetes_readmission")
    
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    
    preprocessor = prepare_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    with mlflow.start_run(run_name="RandomForest"):
        model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        model.fit(X_train_processed, y_train)
        
        y_pred = model.predict(X_val_processed)
        
        f1 = f1_score(y_val, y_pred, average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)
        
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 10)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall_class_2", recall_class_2)
        
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
        
        mlflow.sklearn.log_model(model, "model", input_example=X_train_processed[:5])
        
        run = mlflow.active_run()
        context['task_instance'].xcom_push(key='rf_run_id', value=run.info.run_id)
        context['task_instance'].xcom_push(key='rf_f1_score', value=f1)

def train_xgboost(**context):
    mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("diabetes_readmission")
    
    X_train, y_train = load_data('train')
    X_val, y_val = load_data('val')
    
    preprocessor = prepare_pipeline(X_train)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    
    with mlflow.start_run(run_name="XGBoost"):
        model = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
        model.fit(X_train_processed, y_train)
        
        y_pred = model.predict(X_val_processed)
        
        f1 = f1_score(y_val, y_pred, average='weighted')
        accuracy = accuracy_score(y_val, y_pred)
        recall_class_2 = recall_score(y_val, y_pred, labels=[2], average='macro', zero_division=0)
        
        mlflow.log_param("model_type", "XGBoost")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_metric("f1_score_weighted", f1)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("recall_class_2", recall_class_2)
        
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

        mlflow.xgboost.log_model(model, "model", input_example=X_train_processed[:5])

        run = mlflow.active_run()
        context['task_instance'].xcom_push(key='xgb_run_id', value=run.info.run_id)
        context['task_instance'].xcom_push(key='xgb_f1_score', value=f1)

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
