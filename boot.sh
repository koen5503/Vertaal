#!/usr/bin/env bash
# ==============================================================================
# boot.sh - Self-Healing & Fast Container Bootstrapper for RunPod Mini-Pod
# ==============================================================================
# Executed upon every pod creation / start via RunPod Docker command:
#   bash -c "[ -f /workspace/boot.sh ] && bash /workspace/boot.sh || (curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh)"
#
# Startup target:
# - First run on clean Global Volume: automatically seeds and sets up (<3-5m)
# - Subsequent boots: instant unpack & start in < 15 seconds!
# ==============================================================================

set -euo pipefail

echo "=== [1/8] Checking Essential Utilities ==="
if ! command -v curl >/dev/null 2>&1 || ! command -v ps >/dev/null 2>&1; then
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl procps
fi

echo "=== [2/8] Self-Healing: Syncing Global Volume Files on /workspace ==="
# Ensure /workspace/boot.sh exists
if [ ! -f "/workspace/boot.sh" ]; then
    echo "[Boot] boot.sh missing from /workspace. Copying or downloading..."
    if [ -f "/tmp/boot.sh" ]; then
        cp -f /tmp/boot.sh /workspace/boot.sh
    elif [ -f "$0" ] && [ "$0" != "/workspace/boot.sh" ]; then
        cp -f "$0" /workspace/boot.sh
    else
        curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/boot.sh -o /workspace/boot.sh
    fi
fi

# Always sync latest setup_once.sh and runpod_server.py from repository (never blocked by stale files)
echo "[Boot] Syncing latest setup_once.sh and runpod_server.py from repository..."
curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/setup_once.sh -o /workspace/setup_once.sh
mkdir -p /workspace/app
curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/runpod_server.py -o /workspace/app/runpod_server.py

# Self-Initialization Check: If venv_bundle or models are missing, run automated setup_once.sh
if [ ! -f "/workspace/venv_bundle.tar" ] || [ ! -d "/workspace/models/faster-whisper-large-v3" ] || [ ! -d "/workspace/models/ollama" ]; then
    echo "[Boot] First run detected on /workspace: running automated setup_once.sh..."
    bash /workspace/setup_once.sh
fi

echo "=== [3/8] Verifying Ollama Installation in Container OS ==="
if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama engine into container..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== [4/8] Creating Volatile Local SSD Directories ==="
mkdir -p /root/app /tmp/hf_cache /tmp/torch_cache /root/.ollama/models

echo "=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ==="
if [ -f "/workspace/venv_bundle.tar" ]; then
    tar -xf /workspace/venv_bundle.tar -C /root/
else
    echo "ERROR: /workspace/venv_bundle.tar still not found after setup_once.sh!" >&2
    exit 1
fi

echo "=== [6/8] Syncing Application Code to /root/app ==="
if [ -d "/workspace/app" ]; then
    cp -r /workspace/app/* /root/app/
fi

echo "=== [7/8] Staging Ollama Models & Exporting Environment ==="
if [ -d "/workspace/models/ollama" ] && [ ! -d "/root/.ollama/models/manifests" ]; then
    echo "Staging Ollama models from /workspace to local NVMe SSD..."
    cp -r --no-preserve=mode,ownership /workspace/models/ollama/* /root/.ollama/models/
fi

export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export PYTHONUNBUFFERED=1
export OLLAMA_MODELS=/root/.ollama/models
export WHISPER_MODEL=/workspace/models/faster-whisper-large-v3

# Configure CUDA / cuDNN library paths
for nv_lib in /root/venv/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$nv_lib" ]; then
        export LD_LIBRARY_PATH="${nv_lib}:${LD_LIBRARY_PATH:-}"
    fi
done
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=== [8/8] Starting Services ==="
pkill -9 -f "ollama serve" 2>/dev/null || true
sleep 1
ollama serve &
sleep 3

echo "Launching Live Translation Server on port 8000..."
cd /root/app
exec /root/venv/bin/python3 runpod_server.py
