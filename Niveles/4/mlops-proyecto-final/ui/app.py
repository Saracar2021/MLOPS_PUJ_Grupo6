import streamlit as st
import requests
import pandas as pd
import os
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import shap
import matplotlib.pyplot as plt
import psycopg2
from datetime import datetime
import hashlib  # <--- NUEVO: Para convertir texto a números

# Configuración
st.set_page_config(page_title="Real Estate MLOps Final", page_icon="🏡", layout="wide")
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")

# --- CORRECCIÓN PUERTO BD ---
db_port = os.getenv('POSTGRES_PORT', '5432')
if str(db_port).startswith('tcp://'):
    db_port = '5432'

def get_db_connection():
    return psycopg2.connect(
        host=os.getenv('POSTGRES_HOST', 'postgres'),
        port=db_port,
        database='clean_data',
        user=os.getenv('POSTGRES_USER', 'mlops_user'),
        password=os.getenv('POSTGRES_PASSWORD', 'mlops_password')
    )

# --- NUEVO: FUNCIÓN DE PREPROCESAMIENTO PARA SHAP ---
def preprocess_for_shap(df_input):
    """
    Convierte las columnas de texto a números (Hash) igual que la API,
    para que el modelo y SHAP no fallen por 'feature mismatch'.
    """
    df = df_input.copy()
    
    # Lista exacta de columnas que espera el modelo (orden importante)
    expected_cols = ['bed', 'bath', 'acre_lot', 'house_size', 
                     'brokered_by', 'status', 'city', 'state', 'zip_code']
    
    # Convertir categóricas a Hash
    cat_cols = ['brokered_by', 'status', 'city', 'state', 'zip_code']
    for col in cat_cols:
        if col in df.columns:
            # Hash determinista (String -> Int)
            df[col] = df[col].apply(lambda x: int(hashlib.md5(str(x).encode()).hexdigest(), 16) % 10000)
        else:
            df[col] = 0 # Relleno si falta alguna
            
    # Retornar solo las columnas necesarias y en orden
    return df[expected_cols]

# ================= SIDEBAR =================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/1055/1055646.png", width=100)
    st.header("⚙️ Panel de Control")
    st.markdown("---")
    
    if st.button("🔄 Recargar Modelo en API", type="secondary", use_container_width=True):
        try:
            with st.spinner("Conectando con API..."):
                res = requests.post(f"{FASTAPI_URL}/reload_model", timeout=10)
            
            if res.status_code == 200:
                data = res.json()
                version = data.get('version') or data.get('model_version')
                rmse = data.get('model_rmse')
                
                if version:
                    st.success(f"✅ Modelo Activo: {version}")
                    if rmse:
                        st.caption(f"RMSE: {rmse:.2f}")
                else:
                    st.warning("⚠️ Formato inesperado:")
                    st.json(data)
            else:
                st.error(f"❌ Error {res.status_code}: {res.text}")
        except Exception as e:
            st.error(f"❌ Fallo de conexión: {str(e)}")

# ================= APP PRINCIPAL =================
st.title("🏗️ Proyecto Final MLOps - Grupo 6")
tabs = st.tabs(["🔮 Predicción", "📊 Historial", "🧠 Explicabilidad (SHAP)"])

if 'input_data' not in st.session_state:
    st.session_state.input_data = None

# ... TAB 1: PREDICCIÓN ...
with tabs[0]:
    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Datos de la Propiedad")
        bed = st.number_input("Habitaciones", 1, 10, 3)
        bath = st.number_input("Baños", 1.0, 10.0, 2.0)
        acre_lot = st.number_input("Acres", 0.0, 50.0, 0.15)
        house_size = st.number_input("Tamaño (sqft)", 100.0, 10000.0, 1500.0)
        zip_code = st.text_input("Código Postal", "00601")
        city = st.text_input("Ciudad", "Adjuntas")
        state = st.text_input("Estado", "Puerto Rico")
        status = st.selectbox("Estado", ["for_sale", "sold"])
        brokered_by = st.text_input("Agencia", "Molina")

    with col2:
        st.subheader("Resultado")
        if st.button("Estimar Precio 💰", type="primary", use_container_width=True):
            # Guardamos TODOS los datos (texto incluido)
            payload = {
                "brokered_by": brokered_by, "status": status, "bed": int(bed),
                "bath": float(bath), "acre_lot": float(acre_lot), "city": city,
                "state": state, "zip_code": zip_code, "house_size": float(house_size)
            }
            st.session_state.input_data = pd.DataFrame([payload])
            
            try:
                response = requests.post(f"{FASTAPI_URL}/predict", json=payload)
                if response.status_code == 200:
                    res = response.json()
                    st.success(f"## ${res.get('predicted_price', 0):,.2f}")
                    st.info(f"Modelo: {res.get('model_version')} | RMSE: {res.get('model_rmse'):.2f}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Error de conexión: {e}")

# ... TAB 2: HISTORIAL ...
with tabs[1]:
    st.header("Evolución del Entrenamiento")
    if st.button("🔄 Refrescar Tabla"):
        st.rerun()
    try:
        mlflow.set_tracking_uri(MLFLOW_URL)
        client = MlflowClient()
        experiment = client.get_experiment_by_name("real_estate_prediction")
        if experiment:
            runs = client.search_runs([experiment.experiment_id], order_by=["start_time DESC"])
            data = []
            for run in runs:
                m = run.data.metrics
                ts = run.info.start_time
                fecha = datetime.fromtimestamp(ts/1000).strftime("%Y-%m-%d %H:%M") if ts else "Unknown"
                data.append({
                    "Fecha": fecha,
                    "Modelo": run.data.params.get("model_type"),
                    "RMSE": m.get("rmse"),
                    "R2": m.get("r2"),
                    "Run ID": run.info.run_id
                })
            st.dataframe(pd.DataFrame(data), use_container_width=True)
        else:
            st.warning("No hay experimento registrado.")
    except Exception as e:
        st.error(f"Error MLflow: {e}")

# ... TAB 3: SHAP ...
with tabs[2]:
    st.header("Interpretabilidad del Modelo")
    
    if st.button("📊 Generar Análisis SHAP"):
        with st.spinner("Analizando importancia de variables..."):
            try:
                # 1. Obtener datos de fondo (TRAEMOS TODAS LAS COLUMNAS)
                conn = get_db_connection()
                # Seleccionamos las 9 columnas
                query = """
                    SELECT bed, bath, acre_lot, house_size, 
                           brokered_by, status, city, state, zip_code 
                    FROM clean_data LIMIT 100
                """
                X_bg_raw = pd.read_sql(query, conn)
                conn.close()
                
                # Preprocesar fondo (Texto -> Hash)
                X_bg = preprocess_for_shap(X_bg_raw)
                
                # 2. Cargar modelo
                mlflow.set_tracking_uri(MLFLOW_URL)
                client = MlflowClient()
                experiment = client.get_experiment_by_name("real_estate_prediction")
                runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
                
                if runs:
                    run_id = runs[0].info.run_id
                    model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                    
                    # 3. Calcular SHAP
                    explainer = shap.Explainer(model.predict, X_bg)
                    
                    # Obtener dato del usuario o default
                    if st.session_state.input_data is not None:
                        # Preprocesar input usuario (Texto -> Hash)
                        X_target = preprocess_for_shap(st.session_state.input_data)
                    else:
                        st.warning("⚠️ Usando datos aleatorios (Haz una predicción primero)")
                        X_target = X_bg.iloc[[0]]

                    shap_values = explainer(X_target)

                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("Impacto Local")
                        fig, ax = plt.subplots()
                        shap.plots.waterfall(shap_values[0], show=False)
                        st.pyplot(fig)
                    
                    with col2:
                        st.subheader("Importancia Global")
                        # Beeswarm necesita más datos, usamos el background
                        shap_values_global = explainer(X_bg.head(50))
                        fig2, ax2 = plt.subplots()
                        shap.plots.beeswarm(shap_values_global, show=False)
                        st.pyplot(fig2)
                else:
                    st.error("No se encontró ningún modelo entrenado en MLflow.")
                    
            except Exception as e:
                st.error(f"Error generando SHAP: {str(e)}")
