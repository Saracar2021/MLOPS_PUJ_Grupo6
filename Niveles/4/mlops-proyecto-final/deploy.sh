#!/bin/bash
set -e

echo "🚀 Desplegando MLOps Proyecto Final - Grupo 9"
echo "=============================================="

NAMESPACE="mlops-proyecto-final"

# Verificar que Minikube esté corriendo
if ! minikube status &>/dev/null; then
    echo "❌ Minikube no está corriendo"
    echo "   Ejecuta: minikube start --memory=16384 --cpus=6 --driver=docker"
    exit 1
fi

echo "✅ Minikube está corriendo"

# Crear namespace
echo ""
echo "📦 Creando namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Desplegar componentes en orden
echo ""
echo "1️⃣ Desplegando PostgreSQL..."
kubectl apply -f k8s/postgres/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=postgres -n $NAMESPACE --timeout=300s

echo ""
echo "2️⃣ Desplegando MinIO..."
kubectl apply -f k8s/minio/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=minio -n $NAMESPACE --timeout=300s

echo ""
echo "3️⃣ Desplegando MLflow..."
kubectl apply -f k8s/mlflow/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=mlflow -n $NAMESPACE --timeout=300s

echo ""
echo "4️⃣ Desplegando Airflow..."
kubectl apply -f k8s/airflow/all-in-one.yaml
sleep 30
kubectl wait --for=condition=ready pod -l app=airflow-webserver -n $NAMESPACE --timeout=300s || true

echo ""
echo "5️⃣ Desplegando API FastAPI..."
kubectl apply -f k8s/api/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=fastapi -n $NAMESPACE --timeout=300s

echo ""
echo "6️⃣ Desplegando UI Streamlit..."
kubectl apply -f k8s/ui/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=streamlit -n $NAMESPACE --timeout=300s

echo ""
echo "7️⃣ Desplegando Prometheus..."
kubectl apply -f k8s/prometheus/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=prometheus -n $NAMESPACE --timeout=300s

echo ""
echo "8️⃣ Desplegando Grafana..."
kubectl apply -f k8s/grafana/all-in-one.yaml
kubectl wait --for=condition=ready pod -l app=grafana -n $NAMESPACE --timeout=300s

echo ""
echo "=============================================="
echo "✅ Despliegue completado"
echo "=============================================="
echo ""

echo "📊 URLs de acceso:"
echo ""
echo "Airflow UI:"
minikube service airflow-webserver -n $NAMESPACE --url | head -1
echo "   Usuario: admin / Contraseña: admin"
echo ""
echo "MLflow UI:"
minikube service mlflow -n $NAMESPACE --url | head -1
echo ""
echo "Streamlit UI:"
minikube service streamlit -n $NAMESPACE --url | head -1
echo ""
echo "FastAPI Docs:"
FASTAPI_URL=$(minikube service fastapi -n $NAMESPACE --url | head -1)
echo "$FASTAPI_URL/docs"
echo ""
echo "Grafana:"
minikube service grafana -n $NAMESPACE --url | head -1
echo "   Usuario: admin / Contraseña: admin"
echo ""

echo "=============================================="
echo "🔍 Comandos útiles:"
echo "=============================================="
echo "# Ver todos los pods:"
echo "kubectl get pods -n $NAMESPACE"
echo ""
echo "# Ver logs de Airflow scheduler:"
echo "kubectl logs -f deployment/airflow-scheduler -n $NAMESPACE"
echo ""
echo "# Ver logs de FastAPI:"
echo "kubectl logs -f deployment/fastapi -n $NAMESPACE"
echo ""
