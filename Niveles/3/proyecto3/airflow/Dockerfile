FROM apache/airflow:2.7.0-python3.9

# Switch to root to install system dependencies if needed
USER root

# Install any system dependencies if required
# RUN apt-get update && apt-get install -y <packages> && apt-get clean

# Switch back to airflow user
USER airflow

# Copy requirements file
COPY requirements.txt /requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r /requirements.txt

# Copy DAG files to a temporary location in the image
COPY --chown=airflow:root dags/ /opt/airflow/dags-source/

# Set working directory
WORKDIR /opt/airflow
