#!/bin/bash

set -e

echo "========================================="
echo "Airflow Redeploy Script"
echo "Grupo 6 - Pontificia Universidad Javeriana"
echo "========================================="
echo ""

NAMESPACE="mlops-proyecto3"

# Step 1: Configure Docker to use Minikube's daemon
echo "Step 1: Configuring Docker to use Minikube's daemon..."
eval $(minikube docker-env)

# Step 2: Build the custom Airflow image
echo ""
echo "Step 2: Building custom Airflow image..."
cd airflow
./build-image.sh
cd ..

# Step 3: Delete existing Airflow deployment
echo ""
echo "Step 3: Deleting existing Airflow deployment..."
kubectl delete -f k8s/airflow/all-in-one.yaml --ignore-not-found=true

echo "Waiting for pods to terminate..."
sleep 10

# Step 4: Redeploy Airflow with new configuration
echo ""
echo "Step 4: Deploying Airflow with updated configuration..."
kubectl apply -f k8s/airflow/all-in-one.yaml

echo ""
echo "Step 5: Waiting for Airflow pods to be ready..."
echo "This may take a few minutes..."
sleep 20

kubectl wait --for=condition=ready pod -l app=airflow-webserver -n $NAMESPACE --timeout=300s || echo "Webserver not ready yet, check status manually"
kubectl wait --for=condition=ready pod -l app=airflow-scheduler -n $NAMESPACE --timeout=300s || echo "Scheduler not ready yet, check status manually"

echo ""
echo "========================================="
echo "Redeploy Complete!"
echo "========================================="
echo ""

echo "Airflow UI:"
minikube service airflow-webserver -n $NAMESPACE --url | head -1
echo "  Credentials: admin / admin"
echo ""

echo "========================================="
echo "Verification Commands:"
echo "========================================="
echo "# Check pod status:"
echo "kubectl get pods -n $NAMESPACE | grep airflow"
echo ""
echo "# Check webserver logs:"
echo "kubectl logs -f deployment/airflow-webserver -n $NAMESPACE"
echo ""
echo "# Check scheduler logs:"
echo "kubectl logs -f deployment/airflow-scheduler -n $NAMESPACE"
echo ""
echo "# Check if DAGs were copied (should show your DAG files):"
echo "kubectl exec -it deployment/airflow-webserver -n $NAMESPACE -- ls -la /opt/airflow/dags/"
echo ""
