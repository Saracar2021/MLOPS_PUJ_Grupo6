from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
import requests
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import os
import json
from io import StringIO
from sqlalchemy import create_engine, text
import psycopg2

# Configuración de MLflow
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
DATA_API_URL = os.getenv('DATA_API_URL', 'http://data_api:8080')
GROUP_NUMBER = os.getenv('GROUP_NUMBER', '6')

# Configuración de PostgreSQL para datos
POSTGRES_DATA_URI = "postgresql://data_user:data_password@postgres_data:5432/forest_data"

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
    Tarea 1: Obtener datos de la API externa y guardarlos en PostgreSQL
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
        
        # Obtener datos en formato JSON
        data_json = response.json()
        print(f"Datos recibidos: {data_json.get('group_number')}, Batch: {data_json.get('batch_number')}")
        print(f"Número de registros: {len(data_json.get('data', []))}")
        
        # Guardar en XCom para la siguiente tarea
        kwargs['ti'].xcom_push(key='raw_data_json', value=data_json)

        # NUEVO: Guardar datos en PostgreSQL
        try:
            engine = create_engine(POSTGRES_DATA_URI)
            with engine.begin() as conn:  # CAMBIO: .begin() en lugar de .connect()
                insert_query = text("""
                    INSERT INTO batch_data
                    (group_number, batch_number, execution_date, data_json, row_count)
                    VALUES (:group_num, :batch_num, :exec_date, :data, :count)
                """)
                conn.execute(insert_query, {
                    'group_num': int(GROUP_NUMBER),
                    'batch_num': data_json.get('batch_number'),
                    'exec_date': datetime.now(),
                    'data': json.dumps(data_json),
                    'count': len(data_json.get('data', []))
                })
                # NO NECESITAS conn.commit() - .begin() hace auto-commit
            print(f"✅ Datos guardados en PostgreSQL - Batch {data_json.get('batch_number')}")
        except Exception as e:
            print(f"⚠️ Error al guardar en PostgreSQL: {str(e)}")
            # No fallar el DAG por este error

        return f"Datos obtenidos exitosamente - Batch {data_json.get('batch_number')} - {len(data_json.get('data', []))} registros"
    
    except Exception as e:
        print(f"Error al obtener datos: {str(e)}")
        raise

def get_historical_data(**kwargs):
    """
    Obtener todos los datos históricos de la BD para entrenamiento acumulativo
    """
    try:
        engine = create_engine(POSTGRES_DATA_URI)
        with engine.connect() as conn:
            query = text("""
                SELECT data_json, batch_number, execution_date
                FROM batch_data
                WHERE group_number = :group_num
                ORDER BY execution_date ASC
            """)
            result = conn.execute(query, {'group_num': int(GROUP_NUMBER)})

            all_data = []
            batch_info = []

            for row in result:
                # CAMBIO: PostgreSQL JSONB devuelve dict, no string
                data_obj = row[0] if isinstance(row[0], dict) else json.loads(row[0])
                all_data.extend(data_obj.get('data', []))
                batch_info.append({
                    'batch': row[1],
                    'date': row[2].isoformat() if row[2] else None
                })

            print(f"📊 Datos históricos recuperados: {len(all_data)} registros de {len(batch_info)} batches")
            print(f"Batches incluidos: {[b['batch'] for b in batch_info]}")

            # Guardar en XCom
            kwargs['ti'].xcom_push(key='historical_data', value=all_data)
            kwargs['ti'].xcom_push(key='batch_info', value=batch_info)

            return f"Recuperados {len(all_data)} registros históricos"

    except Exception as e:
        print(f"Error al obtener datos históricos: {str(e)}")
        raise

def preprocess_data(**kwargs):
    """
    Tarea 2: Preprocesar los datos históricos acumulados
    """
    ti = kwargs['ti']

    # Usar datos históricos acumulados en lugar de solo el batch actual
    historical_data = ti.xcom_pull(key='historical_data', task_ids='get_historical_data')
    batch_info = ti.xcom_pull(key='batch_info', task_ids='get_historical_data')

    if not historical_data:
        raise ValueError("No se encontraron datos históricos para procesar")

    print(f"🔄 Entrenamiento ACUMULATIVO con {len(historical_data)} registros")
    print(f"📦 Batches incluidos: {[b['batch'] for b in batch_info]}")
    
    # Definir nombres de columnas según la estructura del dataset
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
    
    # Crear DataFrame desde datos históricos acumulados
    df = pd.DataFrame(historical_data, columns=column_names)
    
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
    
    # Guardar metadata de entrenamiento acumulativo
    ti.xcom_push(key='total_batches', value=len(batch_info))
    ti.xcom_push(key='batch_numbers', value=[b['batch'] for b in batch_info])

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
    
    # Recuperar metadata de entrenamiento acumulativo
    total_batches = ti.xcom_pull(key='total_batches', task_ids='preprocess_data')
    batch_numbers = ti.xcom_pull(key='batch_numbers', task_ids='preprocess_data')
    
    X = pd.read_json(StringIO(X_json))
    y = pd.read_json(StringIO(y_json), typ='series')
    
    print(f"Iniciando entrenamiento con {X.shape[0]} muestras")
    print(f"Total de batches acumulados: {total_batches}")
    print(f"Batches: {batch_numbers}")
    
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
            
            # Registrar parámetros (INCLUYE METADATA ACUMULATIVA)
            mlflow.log_params({
                'model_type': model_name,
                'train_samples': len(X_train),
                'test_samples': len(X_test),
                'n_features': X_train.shape[1],
                'n_classes': len(y.unique()),
                'total_batches': total_batches,  # NUEVO
                'batch_numbers': str(batch_numbers),  # NUEVO
                'cumulative_training': True  # NUEVO
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
    print(f"Columnas del modelo: {X.columns.tolist()}")
    
    return f"Entrenamiento completado. Mejor modelo: {best_model[0]}"


def evaluate_and_register_best(**kwargs):
    """
    Tarea 4: Evaluar resultados, marcar el mejor modelo y PROMOVER A PRODUCTION
    """
    ti = kwargs['ti']
    results_json = ti.xcom_pull(key='training_results', task_ids='train_models')
    results = json.loads(results_json)
    
    print("="*80)
    print("📊 RESULTADOS DE TODOS LOS MODELOS:")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"\n{model_name}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # Identificar el mejor modelo
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    
    print("\n" + "="*80)
    print(f"🏆 MEJOR MODELO: {best_model_name}")
    print(f"📈 Accuracy: {best_accuracy:.4f}")
    print("="*80)
    
    # NUEVO: Promover el mejor modelo a Production
    try:
        client = MlflowClient()
        registered_model_name = f"forest_cover_{best_model_name}"
        
        # Obtener todas las versiones del modelo
        versions = client.search_model_versions(f"name='{registered_model_name}'")
        
        if versions:
            # Ordenar por número de versión (descendente) y tomar la última
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            version_number = latest_version.version
            
            print(f"\n🔄 Promoviendo modelo {registered_model_name} versión {version_number} a PRODUCTION...")
            
            # Archivar versiones anteriores en Production
            for v in versions:
                if v.current_stage == "Production" and v.version != version_number:
                    client.transition_model_version_stage(
                        name=registered_model_name,
                        version=v.version,
                        stage="Archived"
                    )
                    print(f"📦 Versión {v.version} archivada")
            
            # Promover nueva versión a Production
            client.transition_model_version_stage(
                name=registered_model_name,
                version=version_number,
                stage="Production"
            )
            
            print(f"✅ Modelo {registered_model_name} v{version_number} ahora en PRODUCTION")
            
        else:
            print(f"⚠️ No se encontraron versiones registradas de {registered_model_name}")
            
    except Exception as e:
        print(f"❌ Error al promover modelo: {str(e)}")
        # No fallar el DAG por este error
    
    return f"Evaluación completada. Mejor modelo: {best_model_name} (Production)"


# Definir tareas
task_fetch_data = PythonOperator(
    task_id='fetch_data',
    python_callable=fetch_data_from_api,
    dag=dag,
)

task_get_historical = PythonOperator(
    task_id='get_historical_data',
    python_callable=get_historical_data,
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
task_fetch_data >> task_get_historical >> task_preprocess >> task_train >> task_evaluate
