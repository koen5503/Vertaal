#!/usr/bin/env bash
# ==============================================================================
# setup_once.sh - One-Time Initializer for RunPod Mini-Pod + Global Volume
# ==============================================================================
# Can be run manually or triggered automatically by boot.sh upon first boot:
#   bash /workspace/setup_once.sh
#
# Constraints honored:
# - NO symlinks on /workspace (FUSE S3 object storage cannot handle symlinks)
# - Python venv lives on volatile /root and is archived into /workspace/venv_bundle.tar
# - Models downloaded & verified on local NVMe SSD, then copied with --no-preserve=mode,ownership
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

if [ -f "/workspace/models/faster-whisper-large-v3/model.bin" ]; then
    echo "=== [5/8] Faster-Whisper large-v3 already present on /workspace (skipping download) ==="
else
    echo "=== [5/8] Downloading Faster-Whisper large-v3 on local NVMe SSD ==="
    mkdir -p /tmp/whisper_model /workspace/models/faster-whisper-large-v3
    /root/venv/bin/python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Systran/faster-whisper-large-v3', local_dir='/tmp/whisper_model')"
    python3 -c "
import os, shutil
for root, dirs, files in os.walk('/tmp/whisper_model'):
    rel = os.path.relpath(root, '/tmp/whisper_model')
    tdir = os.path.join('/workspace/models/faster-whisper-large-v3', rel) if rel != '.' else '/workspace/models/faster-whisper-large-v3'
    os.makedirs(tdir, exist_ok=True)
    for f in files:
        s = os.path.join(root, f)
        d = os.path.join(tdir, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): continue
        with open(s, 'rb') as fin, open(d, 'wb') as fout: shutil.copyfileobj(fin, fout, length=16*1024*1024)
"
    rm -rf /tmp/whisper_model
fi

if [ -d "/workspace/models/ollama/blobs" ] && [ -n "$(ls -A /workspace/models/ollama/blobs 2>/dev/null)" ] && [ -d "/workspace/models/ollama/manifests" ]; then
    echo "=== [6/8] Ollama aya-expanse:8b already present on /workspace (skipping pull) ==="
else
    echo "=== [6/8] Pulling Ollama aya-expanse:8b on local NVMe SSD and copying to /workspace ==="
    mkdir -p /root/.ollama/models /workspace/models/ollama
    pkill -9 -f "ollama" 2>/dev/null || true
    ollama serve &
    OLLAMA_PID=$!
    sleep 4
    ollama pull aya-expanse:8b
    kill -9 "$OLLAMA_PID" 2>/dev/null || pkill -9 -f "ollama" || true
    wait "$OLLAMA_PID" 2>/dev/null || true
    python3 -c "
import os, shutil
for root, dirs, files in os.walk('/root/.ollama/models'):
    rel = os.path.relpath(root, '/root/.ollama/models')
    tdir = os.path.join('/workspace/models/ollama', rel) if rel != '.' else '/workspace/models/ollama'
    os.makedirs(tdir, exist_ok=True)
    for f in files:
        s = os.path.join(root, f)
        d = os.path.join(tdir, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): continue
        with open(s, 'rb') as fin, open(d, 'wb') as fout: shutil.copyfileobj(fin, fout, length=16*1024*1024)
"
fi

echo "=== [7/8] Archiving /root/venv into /workspace/venv_bundle.tar ==="
tar -cf /workspace/venv_bundle.tar -C /root venv

echo "=== [8/8] Staging Application Code and Boot Scripts into /workspace ==="
mkdir -p /workspace/app

# Stage runpod_server.py
if [ -f "./runpod_server.py" ]; then
    cp -f ./runpod_server.py /workspace/app/
elif [ -f "/workspace/runpod_server.py" ]; then
    cp -f /workspace/runpod_server.py /workspace/app/
fi
if [ ! -f "/workspace/app/runpod_server.py" ]; then
    echo "Downloading runpod_server.py from repository..."
    curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/runpod_server.py -o /workspace/app/runpod_server.py
fi

# Stage boot.sh
if [ -f "./boot.sh" ]; then
    cp -f ./boot.sh /workspace/boot.sh
elif [ -f "/tmp/boot.sh" ]; then
    cp -f /tmp/boot.sh /workspace/boot.sh
fi
if [ ! -f "/workspace/boot.sh" ]; then
    echo "Downloading boot.sh from repository..."
    curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/boot.sh -o /workspace/boot.sh
fi

echo "=== One-time setup completed successfully! ==="
echo "The Global Volume /workspace is now fully prepared."
