import streamlit as st
import requests
import pandas as pd
import os

st.set_page_config(page_title="Real Estate Price Prediction", page_icon="🏠", layout="wide")

FASTAPI_URL = os.getenv("FASTAPI_URL", "http://fastapi:8000")
MLFLOW_URL = os.getenv("MLFLOW_URL", "http://mlflow:5000")

st.title("🏠 Real Estate Price Prediction System")
st.markdown("### Predicción de Precios de Propiedades USA")

# Sidebar con información
with st.sidebar:
    st.header("ℹ️ Información del Sistema")
    
    try:
        response = requests.get(f"{FASTAPI_URL}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            st.success("✅ API: Healthy")
            st.metric("Modelo Activo", health.get('model_version', 'N/A'))
            st.metric("RMSE", f"${health.get('model_rmse', 0):,.0f}")
        else:
            st.error("❌ API: Unhealthy")
    except:
        st.error("❌ API: No disponible")
    
    if st.button("🔄 Recargar Modelo"):
        try:
            response = requests.post(f"{FASTAPI_URL}/reload_model", timeout=10)
            if response.status_code == 200:
                st.success("Modelo recargado exitosamente")
                st.rerun()
            else:
                st.error("Error al recargar modelo")
        except:
            st.error("No se pudo conectar con la API")

# Tabs principales
tab1, tab2 = st.tabs(["📊 Predicción", "📜 Historial de Modelos"])

with tab1:
    st.header("Ingrese los datos de la propiedad")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        brokered_by = st.text_input("ID Corredor", value="12345")
        status = st.selectbox("Estado", ["for_sale", "ready_to_build"])
        bed = st.number_input("Habitaciones", min_value=1, max_value=10, value=3)
    
    with col2:
        bath = st.number_input("Baños", min_value=1.0, max_value=10.0, value=2.0, step=0.5)
        acre_lot = st.number_input("Terreno (acres)", min_value=0.1, max_value=100.0, value=0.5, step=0.1)
        house_size = st.number_input("Tamaño casa (sqft)", min_value=500, max_value=10000, value=1500, step=100)
    
    with col3:
        city = st.text_input("Ciudad", value="Boston")
        state = st.text_input("Estado", value="Massachusetts")
        zip_code = st.text_input("Código Postal", value="02101")
    
    if st.button("🔮 Predecir Precio", type="primary", use_container_width=True):
        payload = {
            "brokered_by": brokered_by,
            "status": status,
            "bed": bed,
            "bath": bath,
            "acre_lot": acre_lot,
            "city": city,
            "state": state,
            "zip_code": zip_code,
            "house_size": house_size
        }
        
        try:
            with st.spinner("Calculando predicción..."):
                response = requests.post(f"{FASTAPI_URL}/predict", json=payload, timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                
                st.success("### Predicción Completada")
                
                # Resultado principal
                col_a, col_b = st.columns([2, 1])
                with col_a:
                    st.metric(
                        "💰 Precio Estimado",
                        f"${result['predicted_price']:,.0f}",
                        delta=None
                    )
                with col_b:
                    st.info(f"**Modelo**: {result['model_version']}")
                    st.info(f"**RMSE**: ${result['model_rmse']:,.0f}")
                
                # Detalles adicionales
                with st.expander("📋 Detalles de la Predicción"):
                    st.json(result)
            else:
                st.error(f"Error en predicción: {response.text}")
        except Exception as e:
            st.error(f"Error de conexión: {str(e)}")

with tab2:
    st.header("Historial de Modelos Entrenados")
    
    try:
        # Intentar obtener información de MLflow
        response = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/get?name=real_estate_model", timeout=10)
        
        if response.status_code == 200:
            model_data = response.json()
            
            st.success("✅ Modelo registrado en MLflow")
            
            # Mostrar versiones
            if 'registered_model' in model_data:
                registered_model = model_data['registered_model']
                
                st.markdown(f"**Nombre**: {registered_model.get('name', 'N/A')}")
                st.markdown(f"**Creado**: {registered_model.get('creation_timestamp', 'N/A')}")
                
                # Tabla de versiones (simplificada)
                st.markdown("### Versiones")
                st.info("Consulta MLflow UI para ver historial completo de versiones y métricas")
                st.markdown(f"[Abrir MLflow UI]({MLFLOW_URL})")
        else:
            st.warning("No se pudo obtener información de modelos")
            st.markdown(f"[Abrir MLflow UI]({MLFLOW_URL})")
    except:
        st.error("Error al conectar con MLflow")
        st.markdown(f"Intenta acceder directamente: {MLFLOW_URL}")

st.markdown("---")
st.caption("MLOps Proyecto Final - Grupo 9 - Pontificia Universidad Javeriana")
