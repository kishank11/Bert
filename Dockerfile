# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    default-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --upgrade pip setuptools wheel

# Install PyTerrier ColBERT from GitHub
RUN pip install --verbose git+https://github.com/terrier-org/pyterrier_colbert.git

# Set environment variables for Java
ENV JAVA_HOME="/usr/lib/jvm/java-11-openjdk-amd64"
ENV PATH="${JAVA_HOME}/bin:${PATH}"

# Copy the rest of the application code into the container
COPY . .

# Specify the default command to run your application
CMD ["python", "main.py"]

