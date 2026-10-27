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

# Self-update boot.sh at the very top to always match repository
REPO_RAW="${RUNPOD_REPO_RAW:-https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline}"
FALLBACK_REPO_RAW="https://raw.githubusercontent.com/koen5503/Vertaal/main"

# Guarded self-update of boot.sh via /tmp
rm -f /tmp/boot_latest.sh
if curl -fsSL "${REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null || curl -fsSL "${FALLBACK_REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null; then
    if [ -s /tmp/boot_latest.sh ] && [ -f /workspace/boot.sh ] && ! cmp -s /tmp/boot_latest.sh /workspace/boot.sh; then
        echo "[Boot] Updating /workspace/boot.sh to latest version from repository..."
        cp -f /tmp/boot_latest.sh /workspace/boot.sh
        rm -f /tmp/boot_latest.sh
        exec bash /workspace/boot.sh
    fi
fi
rm -f /tmp/boot_latest.sh 2>/dev/null || true

# Helper for guarded script syncing via /tmp to protect FUSE storage & survive GitHub outages
sync_repo_file() {
    local file_name="$1"
    local dest_path="$2"
    local tmp_path="/tmp/${file_name%.*}_latest.${file_name##*.}"
    rm -f "$tmp_path"

    # Try primary repository URL first, then fallback
    if (curl -fsSL "${REPO_RAW}/${file_name}" -o "$tmp_path" 2>/dev/null || curl -fsSL "${FALLBACK_REPO_RAW}/${file_name}" -o "$tmp_path" 2>/dev/null) && [ -s "$tmp_path" ]; then
        mkdir -p "$(dirname "$dest_path")"
        cp -f "$tmp_path" "$dest_path"
        rm -f "$tmp_path"
        return 0
    fi
    rm -f "$tmp_path"

    # Guard against GitHub outages: reuse existing file if present and non-empty
    if [ -s "$dest_path" ]; then
        echo "[Boot] Warning: Could not download latest ${file_name} from GitHub, using existing ${dest_path}."
        return 0
    fi

    echo "ERROR: ${dest_path} is missing and could not be downloaded from repository!" >&2
    exit 1
}

echo "=== [1/8] Checking Essential Utilities ==="
if ! command -v curl >/dev/null 2>&1 || ! command -v ps >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl procps zstd ffmpeg libsndfile1
fi

echo "=== [2/8] Self-Healing: Syncing Global Volume Files on /workspace ==="
# Ensure /workspace/boot.sh exists
if [ ! -s "/workspace/boot.sh" ]; then
    echo "[Boot] boot.sh missing or empty on /workspace. Copying or downloading..."
    if [ -s "/tmp/boot.sh" ]; then
        cp -f /tmp/boot.sh /workspace/boot.sh
    elif [ -f "$0" ] && [ "$0" != "/workspace/boot.sh" ] && [ -s "$0" ]; then
        cp -f "$0" /workspace/boot.sh
    else
        sync_repo_file "boot.sh" "/workspace/boot.sh"
    fi
fi

# Always sync latest setup_once.sh and runpod_server.py from repository via /tmp
echo "[Boot] Syncing latest setup_once.sh and runpod_server.py from repository..."
sync_repo_file "setup_once.sh" "/workspace/setup_once.sh"
sync_repo_file "runpod_server.py" "/workspace/app/runpod_server.py"

# Self-Initialization Check: If venv_bundle or models are missing, run automated setup_once.sh
if [ ! -s "/workspace/venv_bundle.tar" ] || [ ! -d "/workspace/models/faster-whisper-large-v3" ] || [ ! -d "/workspace/models/ollama" ]; then
    echo "[Boot] First run detected on /workspace: running automated setup_once.sh..."
    bash /workspace/setup_once.sh
fi

echo "=== [3/8] Verifying Ollama Installation in Container OS ==="
if ! command -v ollama >/dev/null 2>&1; then
    if [ -s "/workspace/ollama_bundle.tar" ]; then
        echo "Restoring Ollama engine from Global Volume cache (1-2s)..."
        tar -xf /workspace/ollama_bundle.tar -C /usr/local/
    else
        echo "Installing Ollama engine into container (first-time install)..."
        curl -fsSL https://ollama.com/install.sh | sh
        echo "Caching Ollama engine to /workspace/ollama_bundle.tar via /tmp..."
        tar -cf /tmp/ollama_bundle.tar -C /usr/local bin/ollama lib/ollama 2>/dev/null || true
        if [ -s "/tmp/ollama_bundle.tar" ]; then
            cp -f /tmp/ollama_bundle.tar /workspace/ollama_bundle.tar
            rm -f /tmp/ollama_bundle.tar
        fi
    fi
fi

echo "=== [4/8] Creating Volatile Local SSD Directories ==="
mkdir -p /root/app /tmp/hf_cache /tmp/torch_cache /root/.ollama/models /root/models

echo "=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ==="
extract_failed=false
if [ -s "/workspace/venv_bundle.tar" ]; then
    tar -xf /workspace/venv_bundle.tar -C /root/ 2>/dev/null || extract_failed=true
else
    extract_failed=true
fi

if [ "$extract_failed" = true ] || [ ! -f "/root/venv/bin/python3" ]; then
    echo "[Boot] Warning: Corrupted or missing /workspace/venv_bundle.tar detected. Self-healing via setup_once.sh..."
    rm -f /workspace/venv_bundle.tar
    bash /workspace/setup_once.sh
fi

if [ ! -f "/root/venv/bin/python3" ]; then
    echo "ERROR: /root/venv/bin/python3 still missing after self-healing setup_once.sh!" >&2
    exit 1
fi

echo "=== [6/8] Syncing Application Code to /root/app ==="
if [ -d "/workspace/app" ]; then
    cp -r /workspace/app/* /root/app/
fi

echo "=== [7/8] Staging Models & Exporting Environment ==="
if [ -d "/workspace/models/ollama" ]; then
    echo "Staging Ollama models from /workspace to local NVMe SSD (/root/.ollama/models)..."
    [ -L /root/.ollama/models ] && rm -f /root/.ollama/models
    mkdir -p /root/.ollama/models
    python3 -c "
import os, shutil
for root, dirs, files in os.walk('/workspace/models/ollama'):
    rel = os.path.relpath(root, '/workspace/models/ollama')
    tdir = os.path.join('/root/.ollama/models', rel) if rel != '.' else '/root/.ollama/models'
    os.makedirs(tdir, exist_ok=True)
    for f in files:
        s = os.path.join(root, f)
        d = os.path.join(tdir, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): continue
        with open(s, 'rb') as fin, open(d, 'wb') as fout: shutil.copyfileobj(fin, fout, length=16*1024*1024)
" 2>/dev/null || true
fi

if [ -d "/workspace/models/faster-whisper-large-v3" ] && [ ! -f "/root/models/faster-whisper-large-v3/model.bin" ]; then
    echo "Staging Faster-Whisper large-v3 from /workspace to local NVMe SSD (/root/models)..."
    python3 -c "
import os, shutil
for root, dirs, files in os.walk('/workspace/models/faster-whisper-large-v3'):
    rel = os.path.relpath(root, '/workspace/models/faster-whisper-large-v3')
    tdir = os.path.join('/root/models/faster-whisper-large-v3', rel) if rel != '.' else '/root/models/faster-whisper-large-v3'
    os.makedirs(tdir, exist_ok=True)
    for f in files:
        s = os.path.join(root, f)
        d = os.path.join(tdir, f)
        if os.path.exists(d) and os.path.getsize(d) == os.path.getsize(s): continue
        with open(s, 'rb') as fin, open(d, 'wb') as fout: shutil.copyfileobj(fin, fout, length=16*1024*1024)
" 2>/dev/null || true
fi

export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export PYTHONUNBUFFERED=1
export OLLAMA_NUM_PARALLEL=2

if [ -d "/root/.ollama/models/manifests" ]; then
    export OLLAMA_MODELS=/root/.ollama/models
else
    export OLLAMA_MODELS=/workspace/models/ollama
fi

if [ -f "/root/models/faster-whisper-large-v3/model.bin" ]; then
    export WHISPER_MODEL=/root/models/faster-whisper-large-v3
else
    export WHISPER_MODEL=/workspace/models/faster-whisper-large-v3
fi

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

echo "Pre-warming Ollama aya-expanse:8b in GPU VRAM..."
ollama run aya-expanse:8b "hi" >/dev/null 2>&1 || true

echo "Launching Live Translation Server on port 8000..."
cd /root/app
exec /root/venv/bin/python3 runpod_server.py
