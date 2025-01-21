FROM python:3.8-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install tira ir-datasets 'python-terrier==0.10.0' 

# Copy script into the container
COPY script.py /app/script.py

# Set the entrypoint to execute the script
ENTRYPOINT ["python3", "/app/script.py"]


