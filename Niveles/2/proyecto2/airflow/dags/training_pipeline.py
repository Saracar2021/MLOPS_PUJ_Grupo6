from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import json
from io import StringIO

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
DATA_API_URL = os.getenv('DATA_API_URL', 'http://data_api:8080')
GROUP_NUMBER = os.getenv('GROUP_NUMBER', '6')

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Argumentos por defecto
default_args = {
    'owner': 'mlops_team',
    'depends_on_past': False,
    'start_date': datetime(2025, 9, 30),
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

# Crear DAG
dag = DAG(
    'forest_cover_training_pipeline',
    default_args=default_args,
    description='Pipeline completo de entrenamiento para Forest Cover Type',
    schedule_interval='*/10 * * * *',  # Cada 10 minutos
    catchup=False,
    tags=['mlops', 'training', 'forest-cover'],
)


def fetch_data_from_api(**kwargs):
    """
    Tarea 1: Obtener datos de la API externa
    """
    print(f"Obteniendo datos de: {DATA_API_URL}")
    print(f"Número de grupo: {GROUP_NUMBER}")
    
    try:
        # Realizar petición a la API
        response = requests.get(
            f"{DATA_API_URL}/data",
            params={'group_number': GROUP_NUMBER},
            timeout=30
        )
        response.raise_for_status()
        
        # CAMBIO: Obtener datos en formato JSON (no CSV)
        data_json = response.json()
        print(f"Datos recibidos: {data_json.get('group_number')}, Batch: {data_json.get('batch_number')}")
        print(f"Número de registros: {len(data_json.get('data', []))}")
        
        # Guardar en XCom para la siguiente tarea
        kwargs['ti'].xcom_push(key='raw_data_json', value=data_json)
        
        return f"Datos obtenidos exitosamente - Batch {data_json.get('batch_number')} - {len(data_json.get('data', []))} registros"
    
    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")
        raise

def preprocess_data(**kwargs):
    """
    Tarea 2: Preprocesar los datos
    """
    ti = kwargs['ti']
    raw_data_json = ti.xcom_pull(key='raw_data_json', task_ids='fetch_data')
    
    if not raw_data_json:
        raise ValueError("No se encontraron datos para procesar")
    
    # CAMBIO: Extraer la lista de datos del JSON
    data_list = raw_data_json.get('data', [])
    
    if not data_list:
        raise ValueError("La respuesta JSON no contiene datos")
    
    print(f"Procesando {len(data_list)} registros del batch {raw_data_json.get('batch_number')}")
    
    # CAMBIO: Definir nombres de columnas según la estructura del dataset
    column_names = [
        'Elevation',
        'Aspect',
        'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am',
        'Hillshade_Noon',
        'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Wilderness_Area',
        'Soil_Type',
        'Cover_Type'
    ]
    
    # CAMBIO: Crear DataFrame desde la lista de listas
    df = pd.DataFrame(data_list, columns=column_names)
    
    print(f"Dataset cargado: {df.shape}")
    print(f"Columnas: {df.columns.tolist()}")
    print(f"Primeras filas:\n{df.head()}")
    
    # Convertir columnas numéricas (vienen como strings)
    numeric_columns = [
        'Elevation', 'Aspect', 'Slope',
        'Horizontal_Distance_To_Hydrology',
        'Vertical_Distance_To_Hydrology',
        'Horizontal_Distance_To_Roadways',
        'Hillshade_9am', 'Hillshade_Noon', 'Hillshade_3pm',
        'Horizontal_Distance_To_Fire_Points',
        'Cover_Type'
    ]
    
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Verificar valores nulos
    print(f"Valores nulos:\n{df.isnull().sum()}")
    
    # Eliminar filas con valores nulos
    df_clean = df.dropna()
    print(f"Dataset después de eliminar nulls: {df_clean.shape}")
    
    # Separar variable objetivo (Cover_Type)
    if 'Cover_Type' not in df_clean.columns:
        raise ValueError("La columna 'Cover_Type' no existe en los datos")
    
    X = df_clean.drop('Cover_Type', axis=1)
    y = df_clean['Cover_Type']
    
    # One-hot encoding para variables categóricas (Wilderness_Area y Soil_Type)
    categorical_columns = ['Wilderness_Area', 'Soil_Type']
    X_encoded = pd.get_dummies(X, columns=categorical_columns, drop_first=False)
    
    print(f"Features después de encoding: {X_encoded.shape}")
    print(f"Distribución de clases:\n{y.value_counts()}")
    
    # Guardar datos procesados
    ti.xcom_push(key='X_processed', value=X_encoded.to_json())
    ti.xcom_push(key='y_processed', value=y.to_json())
    ti.xcom_push(key='feature_names', value=X_encoded.columns.tolist())
    
    return f"Datos preprocesados: {X_encoded.shape[0]} filas, {X_encoded.shape[1]} características"

def train_models(**kwargs):
    """
    Tarea 3: Entrenar múltiples modelos y registrarlos en MLflow
    """
    ti = kwargs['ti']
    
    # Recuperar datos procesados
    X_json = ti.xcom_pull(key='X_processed', task_ids='preprocess_data')
    y_json = ti.xcom_pull(key='y_processed', task_ids='preprocess_data')
    feature_names = ti.xcom_pull(key='feature_names', task_ids='preprocess_data')
    
    X = pd.read_json(StringIO(X_json))
    y = pd.read_json(StringIO(y_json), typ='series')
    
    print(f"Iniciando entrenamiento con {X.shape[0]} muestras")
    
    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Estandarización
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Configurar experimento MLflow
    experiment_name = "forest_cover_classification"
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except:
        experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    mlflow.set_experiment(experiment_name)
    
    # Definir modelos a entrenar
    models = {
        'logistic_regression': LogisticRegression(max_iter=1000, random_state=42),
        'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'gradient_boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    results = {}
    
    # Entrenar cada modelo
    for model_name, model in models.items():
        print(f"\nEntrenando: {model_name}")
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Entrenar
            model.fit(X_train_scaled, y_train)
            
            # Predicciones
            y_pred = model.predict(X_test_scaled)
            
            # Métricas
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # Registrar parámetros
            mlflow.log_params({
                'model_type': model_name,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'n_classes': len(y.unique())
            })
            
            # Registrar métricas
            mlflow.log_metrics({
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            })
            
            # Registrar modelo con el scaler
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=f"forest_cover_{model_name}"
            )
            
            # Guardar scaler por separado
            import joblib
            import tempfile
            with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
                joblib.dump(scaler, f.name)
                mlflow.log_artifact(f.name, artifact_path="scaler")
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall
            }
            
            print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}")
    
    # Guardar resultados
    ti.xcom_push(key='training_results', value=json.dumps(results))
    
    # Encontrar el mejor modelo
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nMejor modelo: {best_model[0]} con accuracy: {best_model[1]['accuracy']:.4f}")
    print(f"Columnas del modelo: {X_encoded.columns.tolist()}")
    
    return f"Entrenamiento completado. Mejor modelo: {best_model[0]}"

def evaluate_and_register_best(**kwargs):
    """
    Tarea 4: Evaluar resultados y marcar el mejor modelo para producción
    """
    ti = kwargs['ti']
    results_json = ti.xcom_pull(key='training_results', task_ids='train_models')
    results = json.loads(results_json)
    
    print("Resultados de todos los modelos:")
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Identificar el mejor modelo
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\nMejor modelo seleccionado: {best_model_name}")
    print(f"Accuracy: {best_accuracy:.4f}")
    
    # Aquí podrías agregar lógica para promover el modelo a producción en MLflow
    # usando mlflow.register_model() con stage='Production'
    
    return f"Evaluación completada. Modelo seleccionado: {best_model_name}"

# Definir tareas
task_fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_api,
    dag=dag,
)

task_preprocess = PythonOperator(
    task_id='preprocess_data',
    python_callable=preprocess_data,
    dag=dag,
)

task_train = PythonOperator(
    task_id='train_models',
    python_callable=train_models,
    dag=dag,
)

task_evaluate = PythonOperator(
    task_id='evaluate_and_register_best',
    python_callable=evaluate_and_register_best,
    dag=dag,
)

# Definir dependencias
task_fetch_data >> task_preprocess >> task_train >> task_evaluate
