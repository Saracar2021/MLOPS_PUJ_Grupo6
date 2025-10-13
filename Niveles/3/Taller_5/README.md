# 🚀 Taller 5 - Locust: Pruebas de Carga y Escalabilidad

**Curso**: MLOps - Pontificia Universidad Javeriana  

## 🎯 Objetivo

Este taller tiene como objetivo implementar y evaluar el rendimiento de una API de inferencia mediante:

1. ✅ Crear una imagen Docker con API FastAPI para inferencia
2. ✅ Publicar la imagen en DockerHub
3. ✅ Configurar docker-compose para deployment
4. ✅ Realizar pruebas de carga con Locust
5. ✅ Limitar recursos y encontrar configuración óptima
6. ✅ Escalar horizontalmente con múltiples replicas
7. ✅ Comparar y documentar resultados

---

### Verificar Instalación

```bash
# Verificar Docker
docker --version
docker-compose --version

# Verificar que Docker está corriendo
docker ps

# Verificar recursos disponibles
docker system info | grep -E "CPUs|Total Memory"
```

## 📁 Estructura del Proyecto

```
taller_locust/
├── README.md                          # Este archivo
├── inference_api/                     # API de Inferencia
│   ├── Dockerfile                     # Imagen Docker de la API
│   ├── main.py                        # Código FastAPI
│   ├── requirements.txt               # Dependencias
│   ├── train_model.py                 # Script para entrenar modelo
│   └── models/                        # Modelos entrenados
│       └── model.pkl                  # Modelo serializado
├── locust_tests/                      # Pruebas de Carga
│   ├── locustfile.py                  # Definición de pruebas
│   └── docker-compose-locust.yml      # Stack Locust completo
├── docker-compose-inference.yml       # API standalone
├── docker-compose-scale.yml           # API con escalamiento
├── nginx.conf                         # Config Load Balancer
└── docs/                              # Documentación
    ├── RESULTADOS.md                  # Resultados de experimentos
    └── screenshots/                   # Capturas de pantalla
```

---

## 🔧 Guía de Instalación

### Paso 1: Clonar el Repositorio

```bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/MLOPS_PUJ_Grupo6.git
cd MLOPS_PUJ_Grupo6/Niveles/5/taller_locust
```

### Paso 2: Entrenar el Modelo

```bash
# Ir al directorio de la API
cd inference_api

# Instalar dependencias (opcional, solo si quieres entrenar local)
pip install -r requirements.txt

# Entrenar modelo
python train_model.py

# Verificar que se creó el modelo
ls -lh models/model.pkl
```

**Salida esperada**:
```
🌲 Entrenando modelo de clasificación...
✅ Datos divididos: 8000 entrenamiento, 2000 prueba
🔨 Entrenando Random Forest Classifier...
✅ Accuracy: 0.9234
💾 Modelo guardado en: models/model.pkl
📦 Tamaño del modelo: 145.32 KB
```

### Paso 3: Construir la Imagen Docker

```bash
# Construir imagen
docker build -t forest-inference:v1 .

# Verificar imagen creada
docker images | grep forest-inference
```

**Salida esperada**:
```
forest-inference   v1     abc123def456   2 minutes ago   245MB
```

### Paso 4: Probar la API Localmente

```bash
# Ejecutar contenedor
docker run -d -p 8989:8989 --name test-api forest-inference:v1

# Probar health check
curl http://localhost:8989/health

# Probar predicción
curl -X POST http://localhost:8989/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Elevation": 2500, "Aspect": 150, "Slope": 10,
    "Horizontal_Distance_To_Hydrology": 300,
    "Vertical_Distance_To_Hydrology": 0,
    "Horizontal_Distance_To_Roadways": 500,
    "Hillshade_9am": 220, "Hillshade_Noon": 230,
    "Hillshade_3pm": 150,
    "Horizontal_Distance_To_Fire_Points": 3000,
    "Wilderness_Area1": 1, "Wilderness_Area2": 0,
    "Wilderness_Area3": 0, "Wilderness_Area4": 0,
    "Soil_Type1": 0, "Soil_Type2": 0, "Soil_Type3": 0,
    "Soil_Type4": 0, "Soil_Type5": 0, "Soil_Type6": 0,
    "Soil_Type7": 0, "Soil_Type8": 0, "Soil_Type9": 0,
    "Soil_Type10": 0, "Soil_Type11": 0, "Soil_Type12": 0,
    "Soil_Type13": 0, "Soil_Type14": 0, "Soil_Type15": 0,
    "Soil_Type16": 0, "Soil_Type17": 0, "Soil_Type18": 0,
    "Soil_Type19": 0, "Soil_Type20": 0, "Soil_Type21": 0,
    "Soil_Type22": 0, "Soil_Type23": 0, "Soil_Type24": 0,
    "Soil_Type25": 0, "Soil_Type26": 0, "Soil_Type27": 0,
    "Soil_Type28": 0, "Soil_Type29": 1, "Soil_Type30": 0,
    "Soil_Type31": 0, "Soil_Type32": 0, "Soil_Type33": 0,
    "Soil_Type34": 0, "Soil_Type35": 0, "Soil_Type36": 0,
    "Soil_Type37": 0, "Soil_Type38": 0, "Soil_Type39": 0,
    "Soil_Type40": 0
  }'

# Detener contenedor de prueba
docker stop test-api && docker rm test-api
```

**Salida esperada**:
```json
{
  "prediction": 2,
  "cover_type": "Lodgepole Pine",
  "confidence": 0.8745,
  "processing_time_ms": 12.34,
  "timestamp": "2025-10-12T10:30:45"
}
```

✅ **Si todo funciona, continúa al siguiente paso**

### Paso 5: Publicar en DockerHub

```bash
# Login en DockerHub
docker login
# Ingresa tu usuario y contraseña

# Tagear imagen con tu usuario
docker tag forest-inference:v1 TU-USUARIO/forest-inference:v1

# Publicar imagen
docker push TU-USUARIO/forest-inference:v1

# Verificar en: https://hub.docker.com/r/TU-USUARIO/forest-inference
```

### Paso 6: Configurar Variables de Entorno

```bash
# Crear archivo .env en la raíz del proyecto
echo "DOCKERHUB_USER=TU-USUARIO" > .env
echo "LOCUST_WORKERS=3" >> .env

# Verificar
cat .env
