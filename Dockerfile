FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Store models inside /root so they are baked permanently into the container image
ENV OLLAMA_MODELS=/root/.ollama/models
ENV HF_HOME=/root/.cache/huggingface

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    zstd \
    procps \
 && rm -rf /var/lib/apt/lists/*
 
# Install Ollama (Native Linux installation to run inside container)
RUN curl -fsSL https://ollama.com/install.sh | sh

# Install Python requirements
RUN pip3 install --no-cache-dir \
    fastapi \
    websockets \
    uvicorn \
    faster-whisper \
    ollama \
    numpy

# Pre-bake Faster-Whisper large-v3 into image (3.1 GB)
RUN python3 -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# Pre-bake Ollama aya-expanse:8b into image (5.1 GB)
RUN ollama serve & \
    sleep 5 && \
    ollama pull aya-expanse:8b && \
    pkill -9 -f "ollama" || true

WORKDIR /app
COPY runpod_server.py /app/runpod_server.py

# Create a startup script to boot Ollama and FastAPI (0s model download required)
RUN echo '#!/bin/bash\n\
# Start ollama daemon in background\n\
ollama serve &\n\
sleep 3\n\
ollama list\n\
\n\
# Boot websocket streaming server\n\
echo "Starting FastAPI PyServer..."\n\
python3 runpod_server.py\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
