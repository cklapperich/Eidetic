FROM python:3.9-slim

# Install basic dependencies and create non-root user
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && useradd -ms /bin/bash vscode

# Set the working directory
WORKDIR /workspace

# Install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER vscode