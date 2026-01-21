# RL Post-training Docker Environment
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /workspace

# System dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# RL 基础库
RUN pip install --no-cache-dir stable-baselines3[extra] gymnasium

# LLM post-training 库
RUN pip install --no-cache-dir \
    transformers \
    trl \
    datasets \
    peft \
    accelerate \
    bitsandbytes

COPY . /workspace
