#!/usr/bin/env bash
# ==============================================================================
# boot.sh - Fast Container Bootstrapper for RunPod Mini-Pod + Global Volume
# ==============================================================================
# Executed upon every pod creation / start via RunPod Docker command:
#   bash -c "bash /workspace/boot.sh"
#
# Startup target: < 15 seconds.
# Unpacks pre-bundled virtual environment to volatile local NVMe SSD (/root/venv)
# and connects to pre-downloaded models on /workspace without symlinks.
# ==============================================================================

set -euo pipefail

echo "=== [1/8] Checking Essential Utilities & Ollama ==="
if ! command -v curl >/dev/null 2>&1 || ! command -v ps >/dev/null 2>&1; then
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl procps
fi

if ! command -v ollama >/dev/null 2>&1; then
    echo "Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

echo "=== [2/8] Creating Volatile Local Directories ==="
mkdir -p /root/app /tmp/hf_cache /tmp/torch_cache

echo "=== [3/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ==="
if [ -f "/workspace/venv_bundle.tar" ]; then
    tar -xf /workspace/venv_bundle.tar -C /root/
else
    echo "ERROR: /workspace/venv_bundle.tar not found! Run setup_once.sh first." >&2
    exit 1
fi

echo "=== [4/8] Syncing Application Code to /root/app ==="
if [ -d "/workspace/app" ]; then
    cp -r /workspace/app/* /root/app/
fi

echo "=== [5/8] Exporting Runtime Environment Variables ==="
export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export PYTHONUNBUFFERED=1
export OLLAMA_MODELS=/workspace/models/ollama
export WHISPER_MODEL=/workspace/models/faster-whisper-large-v3

echo "=== [6/8] Configuring CUDA / cuDNN Library Paths ==="
for nv_lib in /root/venv/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$nv_lib" ]; then
        export LD_LIBRARY_PATH="${nv_lib}:${LD_LIBRARY_PATH:-}"
    fi
done
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

echo "=== [7/8] Starting Ollama Daemon ==="
ollama serve &
sleep 2

echo "=== [8/8] Launching Live Translation Server ==="
cd /root/app
exec /root/venv/bin/python3 runpod_server.py
