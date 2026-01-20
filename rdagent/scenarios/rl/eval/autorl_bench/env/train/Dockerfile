# Base image
FROM python:3.10-slim

# Set working directory
WORKDIR /workspace

# Install dependencies
RUN pip install --no-cache-dir torch torchvision torchaudio
RUN pip install --no-cache-dir stable-baselines3[extra] gymnasium

# Copy the current directory contents into the container at /workspace
COPY . /workspace
