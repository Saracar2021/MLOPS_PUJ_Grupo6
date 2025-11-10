#!/bin/bash

set -e

echo "========================================="
echo "MLOps Proyecto 3 - Deployment Script"
echo "Grupo 6 - Pontificia Universidad Javeriana"
echo "========================================="
echo ""

NAMESPACE="mlops-proyecto3"

echo "Step 1: Checking Minikube status..."
if ! minikube status &>/dev/null; then
    echo "Minikube is not running. Starting Minikube..."
    minikube start --memory=16384 --cpus=6 --driver=docker
else
    echo "Minikube is already running."
fi

echo ""
echo "Step 2: Creating namespace..."
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -
kubectl config set-context --current --namespace=$NAMESPACE

echo ""
echo "Step 3: Deploying PostgreSQL databases..."
echo "  - PostgreSQL RAW..."
kubectl apply -f k8s/postgres-raw/
echo "  - PostgreSQL CLEAN..."
kubectl apply -f k8s/postgres-clean/
echo "  - PostgreSQL MLflow..."
kubectl apply -f k8s/postgres-mlflow/

echo ""
echo "Waiting for PostgreSQL pods to be ready (this may take 2-3 minutes)..."
kubectl wait --for=condition=ready pod -l app=postgres-raw --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=postgres-clean --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=postgres-mlflow --timeout=300s || true

echo ""
echo "Step 4: Deploying MinIO..."
kubectl apply -f k8s/minio/
echo "Waiting for MinIO to be ready..."
kubectl wait --for=condition=ready pod -l app=minio --timeout=300s || true

echo ""
echo "Step 5: Deploying MLflow..."
kubectl apply -f k8s/mlflow/
echo "Waiting for MLflow to be ready..."
kubectl wait --for=condition=ready pod -l app=mlflow --timeout=300s || true

echo ""
echo "Step 6: Deploying Airflow..."
kubectl apply -f k8s/airflow/
echo "Waiting for Airflow to be ready..."
sleep 30
kubectl wait --for=condition=ready pod -l app=airflow-webserver --timeout=300s || true

echo ""
echo "Step 7: Deploying FastAPI..."
kubectl apply -f k8s/fastapi/
echo "Waiting for FastAPI to be ready..."
kubectl wait --for=condition=ready pod -l app=fastapi --timeout=300s || true

echo ""
echo "Step 8: Deploying Streamlit..."
kubectl apply -f k8s/streamlit/
echo "Waiting for Streamlit to be ready..."
kubectl wait --for=condition=ready pod -l app=streamlit --timeout=300s || true

echo ""
echo "Step 9: Deploying Observability (Prometheus + Grafana)..."
kubectl apply -f k8s/prometheus/
kubectl apply -f k8s/grafana/
echo "Waiting for observability services to be ready..."
kubectl wait --for=condition=ready pod -l app=prometheus --timeout=300s || true
kubectl wait --for=condition=ready pod -l app=grafana --timeout=300s || true

echo ""
echo "========================================="
echo "Deployment Complete!"
echo "========================================="
echo ""

echo "Getting service URLs..."
echo ""
echo "Airflow UI:"
minikube service airflow-webserver -n $NAMESPACE --url | head -1
echo "  Credentials: admin / admin"
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
echo "  Credentials: admin / admin"
echo ""

echo "========================================="
echo "Verification Commands:"
echo "========================================="
echo "kubectl get all -n $NAMESPACE"
echo "kubectl logs -f deployment/airflow-scheduler -n $NAMESPACE"
echo "kubectl logs -f deployment/fastapi -n $NAMESPACE"
echo ""

echo "========================================="
echo "Next Steps:"
echo "========================================="
echo "1. Access Airflow UI and activate DAGs"
echo "2. Wait for data ingestion (7 runs x 10min = ~70min)"
echo "3. Run model training DAG"
echo "4. Run model promotion DAG"
echo "5. Test predictions via Streamlit or FastAPI"
echo "6. Run Locust load tests"
echo "7. Check metrics in Grafana"
echo ""
echo "For detailed instructions, see README.md"
echo ""
