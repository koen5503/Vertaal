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

REPO_RAW="${RUNPOD_REPO_RAW:-https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline}"
FALLBACK_REPO_RAW="https://raw.githubusercontent.com/koen5503/Vertaal/main"

echo "=== [1/8] Installing OS Packages ==="
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    zstd curl procps ffmpeg libsndfile1

echo "=== [2/8] Installing Ollama ==="
if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
fi
if [ ! -s "/workspace/ollama_bundle.tar" ] && [ -f "/usr/local/bin/ollama" ]; then
    echo "Caching Ollama binaries to /workspace/ollama_bundle.tar via /tmp..."
    tar -cf /tmp/ollama_bundle.tar -C /usr/local bin/ollama lib/ollama 2>/dev/null || true
    if [ -s "/tmp/ollama_bundle.tar" ]; then
        cp -f /tmp/ollama_bundle.tar /workspace/ollama_bundle.tar
        rm -f /tmp/ollama_bundle.tar
    fi
fi

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

echo "=== [7/8] Archiving /root/venv into /workspace/venv_bundle.tar (via /tmp) ==="
tar -cf /tmp/venv_bundle.tar -C /root venv
if [ -s "/tmp/venv_bundle.tar" ]; then
    cp -f /tmp/venv_bundle.tar /workspace/venv_bundle.tar
    rm -f /tmp/venv_bundle.tar
else
    echo "ERROR: Failed to create non-empty /tmp/venv_bundle.tar!" >&2
    exit 1
fi

echo "=== [8/8] Staging Application Code and Boot Scripts into /workspace ==="
mkdir -p /workspace/app

# Stage runpod_server.py
if [ -s "./runpod_server.py" ]; then
    cp -f ./runpod_server.py /workspace/app/
elif [ -s "/workspace/runpod_server.py" ]; then
    cp -f /workspace/runpod_server.py /workspace/app/
fi
if [ ! -s "/workspace/app/runpod_server.py" ]; then
    echo "Downloading runpod_server.py from repository via /tmp..."
    rm -f /tmp/runpod_server_latest.py
    if (curl -fsSL "${REPO_RAW}/runpod_server.py" -o /tmp/runpod_server_latest.py 2>/dev/null || curl -fsSL "${FALLBACK_REPO_RAW}/runpod_server.py" -o /tmp/runpod_server_latest.py 2>/dev/null) && [ -s "/tmp/runpod_server_latest.py" ]; then
        cp -f /tmp/runpod_server_latest.py /workspace/app/runpod_server.py
        rm -f /tmp/runpod_server_latest.py
    else
        rm -f /tmp/runpod_server_latest.py
        echo "ERROR: Failed to download runpod_server.py to /workspace/app/runpod_server.py!" >&2
        exit 1
    fi
fi

# Stage boot.sh
if [ -s "./boot.sh" ]; then
    cp -f ./boot.sh /workspace/boot.sh
elif [ -s "/tmp/boot.sh" ]; then
    cp -f /tmp/boot.sh /workspace/boot.sh
fi
if [ ! -s "/workspace/boot.sh" ]; then
    echo "Downloading boot.sh from repository via /tmp..."
    rm -f /tmp/boot_latest.sh
    if (curl -fsSL "${REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null || curl -fsSL "${FALLBACK_REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null) && [ -s "/tmp/boot_latest.sh" ]; then
        cp -f /tmp/boot_latest.sh /workspace/boot.sh
        rm -f /tmp/boot_latest.sh
    else
        rm -f /tmp/boot_latest.sh
        echo "ERROR: Failed to download boot.sh to /workspace/boot.sh!" >&2
        exit 1
    fi
fi

echo "=== One-time setup completed successfully! ==="
echo "The Global Volume /workspace is now fully prepared."
