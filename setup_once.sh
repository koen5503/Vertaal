#!/usr/bin/env bash
# ==============================================================================
# setup_once.sh - One-Time Initializer for RunPod Mini-Pod + Global Volume
# ==============================================================================
# Can be run manually or triggered automatically by boot.sh upon first boot:
#   bash /workspace/setup_once.sh
#
# Constraints honored:
# - NO symlinks on /workspace (FUSE S3 object storage cannot handle symlinks)
# - Python venv lives on volatile /root and is archived into /workspace/venv_bundle.tar.zst (compatible with /workspace/venv_bundle.tar)
# - Models downloaded & verified on local NVMe SSD, then copied with --no-preserve=mode,ownership
# ==============================================================================

set -euo pipefail

OLLAMA_VERSION="0.5.7"
BUNDLE_VERSION="2026.10.2+bc79cf0b"
REPO_BRANCH="${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}"
REPO_RAW="${RUNPOD_REPO_RAW:-https://raw.githubusercontent.com/koen5503/Vertaal/${REPO_BRANCH}}"
FALLBACK_REPO_RAW="https://raw.githubusercontent.com/koen5503/Vertaal/main"

# Helper for guarded script syncing via /tmp to protect FUSE storage & survive GitHub outages
sync_repo_file() {
    local file_name="$1"
    local dest_path="$2"
    local safe_name="${file_name//\//_}"
    local tmp_path="/tmp/${safe_name%.*}_latest.${safe_name##*.}"
    rm -f "$tmp_path"

    # Local filesystem fallback first if running locally in repo
    if [ -s "./${file_name}" ]; then
        mkdir -p "$(dirname "$dest_path")"
        cp -f "./${file_name}" "$dest_path"
        return 0
    fi

    # Try primary repository URL first, then fallback
    if { curl -fsSL "${REPO_RAW}/${file_name}" -o "$tmp_path" 2>/dev/null || { [ "$REPO_BRANCH" != "main" ] && curl -fsSL "${FALLBACK_REPO_RAW}/${file_name}" -o "$tmp_path" 2>/dev/null; }; } && [ -s "$tmp_path" ]; then
        mkdir -p "$(dirname "$dest_path")"
        cp -f "$tmp_path" "$dest_path"
        rm -f "$tmp_path"
        return 0
    fi
    rm -f "$tmp_path"

    # Guard against GitHub outages: reuse existing file if present and non-empty
    if [ -s "$dest_path" ]; then
        echo "[Setup] Warning: Could not download latest ${file_name} from GitHub, using existing ${dest_path}."
        return 0
    fi

    echo "ERROR: ${dest_path} is missing and could not be downloaded from repository!" >&2
    exit 1
}

# Concurrency lock on /workspace to prevent multi-pod collisions on global volumes
LOCK_DIR="/workspace/.setup.lock"

acquire_setup_lock() {
    local max_wait=600
    local wait_time=0
    local poll_interval=5

    while true; do
        if mkdir "$LOCK_DIR" 2>/dev/null; then
            echo "$$ $(date +%s) ${RUNPOD_POD_ID:-${HOSTNAME:-unknown}}" > "$LOCK_DIR/owner"
            trap 'rm -rf "$LOCK_DIR"' EXIT INT TERM
            return 0
        fi

        # Lock directory exists: inspect timestamp inside owner file
        local lock_ts
        lock_ts=$(awk '{print $2}' "$LOCK_DIR/owner" 2>/dev/null || echo "0")
        if ! [[ "$lock_ts" =~ ^[0-9]+$ ]]; then
            lock_ts=0
        fi

        local now
        now=$(date +%s)
        local age=$(( now - lock_ts ))

        if [ "$lock_ts" -gt 0 ] && [ "$age" -ge 0 ] && [ "$age" -lt 900 ]; then
            echo "[Setup] Setup lock held by another process (age: ${age}s). Waiting for completion... (${wait_time}/${max_wait}s)"
            sleep "$poll_interval"
            wait_time=$(( wait_time + poll_interval ))
            if [ "$wait_time" -ge "$max_wait" ]; then
                echo "ERROR: Timed out waiting for setup lock after ${wait_time}s!" >&2
                exit 1
            fi

            # Check if setup completed while we were waiting
            if [ ! -d "$LOCK_DIR" ]; then
                if [ -s "/workspace/venv_bundle.tar.zst" ] || [ -s "/workspace/venv_bundle.tar" ] || [ -f "/workspace/.bundle_version" ]; then
                    echo "[Setup] Setup was completed by concurrent process. Skipping duplicate setup."
                    exit 0
                fi
            fi
        else
            echo "[Setup] Stale setup lock detected (>900s old or unreadable). Removing stale lock..."
            rm -rf "$LOCK_DIR"
        fi
    done
}

acquire_setup_lock

echo "=== [1/8] Installing OS Packages ==="
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    zstd curl procps ffmpeg libsndfile1

echo "=== [2/8] Installing Ollama ==="
if [ -s "/workspace/ollama_bundle.tar.zst" ]; then
    echo "Restoring Ollama engine from cached /workspace/ollama_bundle.tar.zst..."
    tar -I zstd -xf /workspace/ollama_bundle.tar.zst -C /usr/local/
elif [ -s "/workspace/ollama_bundle.tar" ]; then
    echo "Restoring Ollama engine from cached /workspace/ollama_bundle.tar..."
    tar -xf /workspace/ollama_bundle.tar -C /usr/local/
else
    echo "Installing Ollama engine into container (pinned version ${OLLAMA_VERSION})..."
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
    tar -I "zstd -1 -T0" -cf /tmp/ollama_bundle.tar.zst -C /usr/local bin/ollama lib/ollama 2>/dev/null || true
    if [ -s "/tmp/ollama_bundle.tar.zst" ]; then
        cp -f /tmp/ollama_bundle.tar.zst /workspace/ollama_bundle.tar.zst.tmp
        if zstd -t /workspace/ollama_bundle.tar.zst.tmp >/dev/null 2>&1; then
            mv -f /workspace/ollama_bundle.tar.zst.tmp /workspace/ollama_bundle.tar.zst
            rm -f /workspace/ollama_bundle.tar
        fi
        rm -f /workspace/ollama_bundle.tar.zst.tmp /tmp/ollama_bundle.tar.zst
    fi
fi

# Verify version matches $OLLAMA_VERSION, reinstalling and recreating cache on mismatch
INSTALLED_OLLAMA_VER=""
if command -v ollama >/dev/null 2>&1; then
    INSTALLED_OLLAMA_VER=$(ollama --version 2>&1 || echo "")
fi

if [ -z "$INSTALLED_OLLAMA_VER" ] || [[ "$INSTALLED_OLLAMA_VER" != *"$OLLAMA_VERSION"* ]]; then
    echo "[Setup] Warning: Ollama version mismatch or missing ('${INSTALLED_OLLAMA_VER}' vs required '${OLLAMA_VERSION}'). Reinstalling pinned version..."
    curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
    echo "Updating cached Ollama engine to /workspace/ollama_bundle.tar.zst via /tmp..."
    tar -I "zstd -1 -T0" -cf /tmp/ollama_bundle.tar.zst -C /usr/local bin/ollama lib/ollama 2>/dev/null || true
    if [ -s "/tmp/ollama_bundle.tar.zst" ]; then
        cp -f /tmp/ollama_bundle.tar.zst /workspace/ollama_bundle.tar.zst.tmp
        if zstd -t /workspace/ollama_bundle.tar.zst.tmp >/dev/null 2>&1; then
            mv -f /workspace/ollama_bundle.tar.zst.tmp /workspace/ollama_bundle.tar.zst
            rm -f /workspace/ollama_bundle.tar
        fi
        rm -f /workspace/ollama_bundle.tar.zst.tmp /tmp/ollama_bundle.tar.zst
    fi
fi

echo "[Setup] Verified Ollama version: $(ollama --version 2>&1 || echo 'unknown')"

echo "=== [3/8] Creating Python Virtual Environment in /root/venv ==="
# Retain access to pre-installed base PyTorch, CUDA, and cuDNN libraries from devel container
python3 -m venv --system-site-packages /root/venv

# Sync requirements-pod.txt before pip install
sync_repo_file "requirements-pod.txt" "/workspace/app/requirements-pod.txt"

echo "=== [4/8] Installing Python Packages in /root/venv ==="
if [ ! -s "/workspace/app/requirements-pod.txt" ]; then
    echo "ERROR: /workspace/app/requirements-pod.txt is missing or empty!" >&2
    exit 1
fi
/root/venv/bin/pip install --no-cache-dir -r /workspace/app/requirements-pod.txt
echo "[Setup] Installed Python packages (pip freeze):"
/root/venv/bin/pip freeze

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

echo "=== [7/8] Archiving /root/venv into /workspace/venv_bundle.tar.zst (via /tmp) ==="
# Creates /workspace/venv_bundle.tar.zst (compatible with /workspace/venv_bundle.tar)
tar -I "zstd -1 -T0" -cf /tmp/venv_bundle.tar.zst -C /root venv
if [ -s "/tmp/venv_bundle.tar.zst" ]; then
    cp -f /tmp/venv_bundle.tar.zst /workspace/venv_bundle.tar.zst.tmp
    if zstd -t /workspace/venv_bundle.tar.zst.tmp >/dev/null 2>&1; then
        mv -f /workspace/venv_bundle.tar.zst.tmp /workspace/venv_bundle.tar.zst
        rm -f /workspace/venv_bundle.tar
        echo "$BUNDLE_VERSION" > /workspace/.bundle_version
    else
        echo "ERROR: Corrupted archive detected during zstd integrity verification!" >&2
        rm -f /workspace/venv_bundle.tar.zst.tmp
        exit 1
    fi
    rm -f /workspace/venv_bundle.tar.zst.tmp /tmp/venv_bundle.tar.zst
else
    echo "ERROR: Failed to create non-empty /tmp/venv_bundle.tar.zst!" >&2
    exit 1
fi

echo "=== [8/8] Staging Application Code and Boot Scripts into /workspace ==="
mkdir -p /workspace/app /workspace/app/core /workspace/app/core/stt

# Stage runpod_server.py
sync_repo_file "runpod_server.py" "/workspace/app/runpod_server.py"

# Stage requirements-pod.txt
sync_repo_file "requirements-pod.txt" "/workspace/app/requirements-pod.txt"

# Stage boot.sh
sync_repo_file "boot.sh" "/workspace/boot.sh"

# Stage core package
for core_file in core/__init__.py core/engine.py core/translation.py core/audio.py core/utils.py core/stt/__init__.py core/stt/base.py core/stt/cloud_engine.py core/stt/local_engine.py core/stt/runpod_engine.py; do
    sync_repo_file "$core_file" "/workspace/app/$core_file"
done

echo "=== One-time setup completed successfully! ==="
echo "The Global Volume /workspace is now fully prepared."
