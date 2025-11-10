#!/bin/bash

# Script to build custom Airflow image with MLOps dependencies

set -e

# Configuration
IMAGE_NAME="custom-airflow"
IMAGE_TAG="2.7.0-mlops"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Optional: Set your Docker registry here
# REGISTRY="your-registry.com"
# FULL_IMAGE_NAME="${REGISTRY}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "Building custom Airflow image: ${FULL_IMAGE_NAME}"

# Build the image
docker build -t ${FULL_IMAGE_NAME} -f Dockerfile .

echo "Image built successfully: ${FULL_IMAGE_NAME}"

# Optional: Push to registry
# echo "Pushing image to registry..."
# docker push ${FULL_IMAGE_NAME}

echo "Done! You can now update the Kubernetes deployment to use: ${FULL_IMAGE_NAME}"
