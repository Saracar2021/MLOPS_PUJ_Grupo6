"""
Script para entrenar un modelo simple de clasificación
y guardarlo para la API de inferencia

Ejecutar: python train_model.py
"""

import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os

print("🌲 Entrenando modelo de clasificación de cobertura forestal...")

# ===============================================
# 1. GENERAR DATOS SINTÉTICOS
# ===============================================
# En un caso real, cargarías datos de un CSV o API
# Para este taller, generamos datos sintéticos

np.random.seed(42)
n_samples = 10000

print(f"📊 Generando {n_samples} muestras sintéticas...")

# Generar features
X = np.random.rand(n_samples, 54)

# Modificar features para simular patrones reales
# Elevación (0-10): afecta fuertemente el tipo de cobertura
X[:, 0] = X[:, 0] * 4000 + 1000  # Elevación entre 1000-5000m

# Generar labels basados en patrones simples
y = np.zeros(n_samples, dtype=int)
for i in range(n_samples):
    elevation = X[i, 0]
    if elevation < 2000:
        y[i] = 1  # Spruce/Fir a baja altitud
    elif elevation < 2500:
        y[i] = 2  # Lodgepole Pine
    elif elevation < 3000:
        y[i] = 3  # Ponderosa Pine
    elif elevation < 3500:
        y[i] = 4  # Cottonwood/Willow
    elif elevation < 4000:
        y[i] = 5  # Aspen
    elif elevation < 4500:
        y[i] = 6  # Douglas-fir
    else:
        y[i] = 7  # Krummholz a gran altitud

# Añadir ruido para hacer el problema más realista
noise_indices = np.random.choice(n_samples, size=int(n_samples * 0.1), replace=False)
y[noise_indices] = np.random.randint(1, 8, len(noise_indices))

# ===============================================
# 2. DIVIDIR DATOS
# ===============================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✅ Datos divididos: {len(X_train)} entrenamiento, {len(X_test)} prueba")

# ===============================================
# 3. ENTRENAR MODELO
# ===============================================
print("🔨 Entrenando Random Forest Classifier...")

model = RandomForestClassifier(
    n_estimators=50,      # Menos árboles para inferencia más rápida
    max_depth=10,         # Limitar profundidad
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ===============================================
# 4. EVALUAR MODELO
# ===============================================
print("📈 Evaluando modelo...")

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n✅ Accuracy: {accuracy:.4f}")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred))

# ===============================================
# 5. GUARDAR MODELO
# ===============================================
# Crear directorio si no existe
os.makedirs("models", exist_ok=True)

model_path = "models/model.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model, f)

print(f"\n💾 Modelo guardado en: {model_path}")
print(f"📦 Tamaño del modelo: {os.path.getsize(model_path) / 1024:.2f} KB")

# ===============================================
# 6. VALIDAR CARGA DEL MODELO
# ===============================================
print("\n🔍 Validando carga del modelo...")

with open(model_path, "rb") as f:
    loaded_model = pickle.load(f)

# Hacer predicción de prueba
sample = X_test[0].reshape(1, -1)
prediction = loaded_model.predict(sample)[0]
print(f"✅ Predicción de prueba: Clase {prediction}")

print("\n🎉 ¡Modelo entrenado y guardado exitosamente!")
print("\n📝 Próximos pasos:")
print("   1. Copiar el archivo models/model.pkl al directorio inference_api/models/")
print("   2. Construir la imagen Docker: docker build -t forest-inference:v1 .")
print("   3. Ejecutar la API: docker run -p 8989:8989 forest-inference:v1")
