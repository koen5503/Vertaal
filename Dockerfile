FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Route large AI model downloads to the /workspace directory (for Network Volume persistence)
ENV OLLAMA_MODELS=/workspace/ollama
ENV HF_HOME=/workspace/huggingface

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    curl \
    git \
    zstd \
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

WORKDIR /app
COPY runpod_server.py /app/runpod_server.py

# Create a startup script to boot Ollama, fetch model, then boot FastAPI
RUN echo '#!/bin/bash\n\
# Start ollama daemon in background\n\
ollama serve &\n\
sleep 5\n\
# Pre-pull the language models\n\
echo "Pre-fetching LLM Models..."\n\
ollama pull ${OLLAMA_MODEL:-aya-expanse:8b}\n\
\n\
# Boot websocket streaming server\n\
echo "Starting FastAPI PyServer..."\n\
python3 runpod_server.py\n\
' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
