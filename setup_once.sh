#!/usr/bin/env bash
# ==============================================================================
# setup_once.sh - One-Time Initializer for RunPod Mini-Pod + Global Volume
# ==============================================================================
# Run this script ONCE inside a temporary RunPod pod with /workspace mounted:
#   bash /workspace/setup_once.sh
#
# Constraints honored:
# - NO symlinks on /workspace (FUSE S3 object storage cannot handle symlinks)
# - Python venv lives on volatile /root and is archived into /workspace/venv_bundle.tar
# - Models stored flat in /workspace/models/
# ==============================================================================

set -euo pipefail

echo "=== [1/8] Installing OS Packages ==="
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    zstd curl procps ffmpeg libsndfile1

echo "=== [2/8] Installing Ollama ==="
curl -fsSL https://ollama.com/install.sh | sh

echo "=== [3/8] Creating Python Virtual Environment in /root/venv ==="
python3 -m venv --system-site-packages /root/venv

echo "=== [4/8] Installing Python Packages in /root/venv ==="
/root/venv/bin/pip install --no-cache-dir \
    fastapi uvicorn websockets faster-whisper ollama numpy huggingface_hub

echo "=== [5/8] Downloading Faster-Whisper large-v3 flat to /workspace/models ==="
mkdir -p /workspace/models/faster-whisper-large-v3
/root/venv/bin/huggingface-cli download Systran/faster-whisper-large-v3 \
    --local-dir /workspace/models/faster-whisper-large-v3 \
    --local-dir-use-symlinks False

echo "=== [6/8] Pulling Ollama aya-expanse:8b model to /workspace/models/ollama ==="
mkdir -p /workspace/models/ollama
export OLLAMA_MODELS=/workspace/models/ollama
ollama serve &
OLLAMA_PID=$!
sleep 5
ollama pull aya-expanse:8b
kill -9 "$OLLAMA_PID" 2>/dev/null || pkill -9 -f "ollama" || true
wait "$OLLAMA_PID" 2>/dev/null || true

echo "=== [7/8] Archiving /root/venv into /workspace/venv_bundle.tar ==="
tar -cf /workspace/venv_bundle.tar -C /root venv

echo "=== [8/8] Staging Application Code into /workspace/app ==="
mkdir -p /workspace/app
if [ -f "./runpod_server.py" ]; then
    cp -f ./runpod_server.py /workspace/app/
elif [ -f "/workspace/runpod_server.py" ]; then
    cp -f /workspace/runpod_server.py /workspace/app/
fi

echo "=== One-time setup completed successfully! ==="
echo "The Global Volume /workspace is now fully prepared."
