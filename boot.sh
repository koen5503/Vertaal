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
OLLAMA_VERSION="0.5.7"
BUNDLE_VERSION="2026.10.2+bc79cf0b"
REPO_BRANCH="${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}"
REPO_RAW="${RUNPOD_REPO_RAW:-https://raw.githubusercontent.com/koen5503/Vertaal/${REPO_BRANCH}}"
FALLBACK_REPO_RAW="https://raw.githubusercontent.com/koen5503/Vertaal/main"

# Guarded self-update of boot.sh via /tmp
rm -f /tmp/boot_latest.sh
if curl -fsSL "${REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null || { [ "$REPO_BRANCH" != "main" ] && curl -fsSL "${FALLBACK_REPO_RAW}/boot.sh" -o /tmp/boot_latest.sh 2>/dev/null; }; then
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
    local safe_name="${file_name//\//_}"
    local tmp_path="/tmp/${safe_name%.*}_latest.${safe_name##*.}"
    rm -f "$tmp_path"

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
        echo "[Boot] Warning: Could not download latest ${file_name} from GitHub, using existing ${dest_path}."
        return 0
    fi

    echo "ERROR: ${dest_path} is missing and could not be downloaded from repository!" >&2
    exit 1
}

BOOT_START_TIME=$(date +%s)
LAST_PHASE_TIME=$BOOT_START_TIME

echo "=== [1/8] Checking Essential Utilities ==="
if [ -x "/workspace/bin/zstd" ] && ! command -v zstd >/dev/null 2>&1; then
    mkdir -p /usr/local/bin
    ln -sf /workspace/bin/zstd /usr/local/bin/zstd
fi
if ! command -v curl >/dev/null 2>&1 || ! command -v ps >/dev/null 2>&1 || ! command -v zstd >/dev/null 2>&1 || ! command -v ffmpeg >/dev/null 2>&1; then
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends curl procps zstd ffmpeg libsndfile1
fi
now=$(date +%s)
echo "[Boot] Phase [1/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

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

# Always sync latest setup_once.sh, runpod_server.py, and requirements-pod.txt from repository via /tmp
echo "[Boot] Syncing latest setup_once.sh, runpod_server.py, and requirements-pod.txt from repository in parallel..."
sync_pids=()
sync_repo_file "setup_once.sh" "/workspace/setup_once.sh" & sync_pids+=($!)
sync_repo_file "runpod_server.py" "/workspace/app/runpod_server.py" & sync_pids+=($!)
sync_repo_file "requirements-pod.txt" "/workspace/app/requirements-pod.txt" & sync_pids+=($!)

mkdir -p /workspace/app/core /workspace/app/core/stt
for core_file in core/__init__.py core/engine.py core/translation.py core/audio.py core/utils.py core/stt/__init__.py core/stt/base.py core/stt/cloud_engine.py core/stt/local_engine.py core/stt/runpod_engine.py; do
    sync_repo_file "$core_file" "/workspace/app/$core_file" & sync_pids+=($!)
done
for spid in "${sync_pids[@]}"; do
    wait "$spid"
done

# Circuit Breaker: prevent runaway cloud billing on consecutive setup failures
run_setup_once() {
    local fail_count
    fail_count=$(cat /workspace/.setup_failures 2>/dev/null || echo "0")
    if ! [[ "$fail_count" =~ ^[0-9]+$ ]]; then
        fail_count=0
    fi

    if [ "$fail_count" -ge 3 ]; then
        echo "[Boot] FATAL: setup_once.sh has failed $fail_count consecutive times. Halting pod to prevent infinite restart billing loop!" >&2
        cat <<EOF > /workspace/setup_fatal.log
FATAL: setup_once.sh failed $fail_count consecutive times on $(date -u 2>/dev/null || date)
Pod ID: ${RUNPOD_POD_ID:-unknown}
Bundle version: ${BUNDLE_VERSION}
Halting container execution to prevent infinite restart loop and runaway billing.
Inspect this volume or clear /workspace/.setup_failures after resolving root cause.
EOF
        if [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
            echo "[Boot] Attempting self-termination of pod ${RUNPOD_POD_ID} via RunPod REST API..." >&2
            curl -fsSL -X DELETE \
                -H "Authorization: Bearer ${RUNPOD_API_KEY}" \
                "https://api.runpod.io/v2/pods/${RUNPOD_POD_ID}" >/dev/null 2>&1 || true
        fi
        echo "[Boot] Entering idle hold loop to stop restart cycles (sleep 86400)..." >&2
        while true; do
            sleep 86400 || true
        done
    fi

    if bash /workspace/setup_once.sh; then
        rm -f /workspace/.setup_failures
    else
        fail_count=$(( fail_count + 1 ))
        echo "$fail_count" > /workspace/.setup_failures
        echo "ERROR: setup_once.sh failed (failure count: $fail_count)." >&2
        exit 1
    fi
}

# Concurrency lock check: if another pod is currently running setup_once.sh on this volume, wait for it
if [ -d "/workspace/.setup.lock" ]; then
    LOCK_TS=$(awk '{print $2}' /workspace/.setup.lock/owner 2>/dev/null || echo "0")
    if ! [[ "$LOCK_TS" =~ ^[0-9]+$ ]]; then LOCK_TS=0; fi
    NOW=$(date +%s)
    AGE=$(( NOW - LOCK_TS ))
    if [ "$LOCK_TS" -gt 0 ] && [ "$AGE" -ge 0 ] && [ "$AGE" -lt 900 ]; then
        echo "[Boot] Another pod is currently running setup on /workspace (lock age: ${AGE}s). Waiting up to 10m for completion..."
        WAIT_TOTAL=0
        while [ -d "/workspace/.setup.lock" ] && [ "$WAIT_TOTAL" -lt 600 ]; do
            sleep 5
            WAIT_TOTAL=$(( WAIT_TOTAL + 5 ))
        done
        if [ -s "/workspace/venv_bundle.tar.zst" ] || [ -f "/workspace/.bundle_version" ]; then
            echo "[Boot] Concurrent setup finished successfully. Proceeding to boot."
        fi
    elif [ "$AGE" -ge 900 ] || [ "$LOCK_TS" -eq 0 ]; then
        echo "[Boot] Removing stale setup lock (>900s old)..."
        rm -rf /workspace/.setup.lock
    fi
fi

# Self-Initialization Check: If venv_bundle or models are missing, run automated setup_once.sh
if { [ ! -s "/workspace/venv_bundle.tar.zst" ] && [ ! -s "/workspace/venv_bundle.tar" ]; } || [ ! -d "/workspace/models/faster-whisper-large-v3" ] || [ ! -d "/workspace/models/ollama" ]; then
    echo "[Boot] First run detected on /workspace: running automated setup_once.sh..."
    run_setup_once
fi

# Bundle Version Manifest Check:
# Case A: Legacy volume adoption - bundle exists but .bundle_version is missing
if [ ! -f /workspace/.bundle_version ] && { [ -s "/workspace/venv_bundle.tar.zst" ] || [ -s "/workspace/venv_bundle.tar" ]; }; then
    echo "[Boot] Adopting BUNDLE_VERSION=${BUNDLE_VERSION} for legacy venv_bundle on /workspace..."
    echo "$BUNDLE_VERSION" > /workspace/.bundle_version
fi

# Case B: Intentional version upgrade - manifest exists and differs from BUNDLE_VERSION
FOUND_BUNDLE_VER=$(cat /workspace/.bundle_version 2>/dev/null || echo "")
if [ -n "$FOUND_BUNDLE_VER" ] && [ "$FOUND_BUNDLE_VER" != "$BUNDLE_VERSION" ]; then
    echo "[Boot] Intentional bundle upgrade detected (volume version '${FOUND_BUNDLE_VER}' vs required version '${BUNDLE_VERSION}'). Running setup_once.sh..."
    run_setup_once
fi

echo "=== [3/8] Verifying Ollama Installation in Container OS ==="
OLLAMA_EXTRACT_PID=""
if ! command -v ollama >/dev/null 2>&1; then
    if [ -s "/workspace/ollama_bundle.tar.zst" ]; then
        echo "Restoring Ollama engine from Global Volume cache (1-2s)..."
        tar -I zstd -xf /workspace/ollama_bundle.tar.zst -C /usr/local/ &
        OLLAMA_EXTRACT_PID=$!
    elif [ -s "/workspace/ollama_bundle.tar" ]; then
        echo "Restoring Ollama engine from Global Volume cache (1-2s)..."
        tar -xf /workspace/ollama_bundle.tar -C /usr/local/ &
        OLLAMA_EXTRACT_PID=$!
    else
        echo "Installing Ollama engine into container (pinned version ${OLLAMA_VERSION})..."
        curl -fsSL https://ollama.com/install.sh | OLLAMA_VERSION="$OLLAMA_VERSION" sh
        echo "Caching Ollama engine to /workspace/ollama_bundle.tar.zst via /tmp..."
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
fi

now=$(date +%s)
echo "[Boot] Phase [3/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "=== [4/8] Creating Volatile Local SSD Directories ==="
find /workspace -maxdepth 1 -name "*.tmp" -mmin +30 -delete 2>/dev/null || true
mkdir -p /root/app /tmp/hf_cache /tmp/torch_cache /root/.ollama /root/models

now=$(date +%s)
echo "[Boot] Phase [4/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ==="
extract_failed=false
if [ -s "/workspace/venv_bundle.tar.zst" ]; then
    tar -I zstd -xf /workspace/venv_bundle.tar.zst -C /root/ 2>/dev/null || extract_failed=true
elif [ -s "/workspace/venv_bundle.tar" ]; then
    tar -xf /workspace/venv_bundle.tar -C /root/ 2>/dev/null || extract_failed=true
else
    extract_failed=true
fi

if [ "$extract_failed" = true ] || [ ! -f "/root/venv/bin/python3" ]; then
    echo "[Boot] Warning: Corrupted or missing /workspace/venv_bundle.tar detected. Self-healing via setup_once.sh..."
    rm -f /workspace/venv_bundle.tar.zst /workspace/venv_bundle.tar
    run_setup_once
fi

SYS_PY_VER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
VENV_PY_VER=""
if [ -x /root/venv/bin/python3 ]; then
    VENV_PY_VER=$(/root/venv/bin/python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || echo "")
fi
if [ -n "$SYS_PY_VER" ] && [ -n "$VENV_PY_VER" ] && [ "$SYS_PY_VER" != "$VENV_PY_VER" ]; then
    echo "[Boot] Warning: Python minor version difference detected: container=$SYS_PY_VER vs venv=$VENV_PY_VER."
    if ! /root/venv/bin/python3 -c 'import fastapi, uvicorn, faster_whisper, ollama' >/dev/null 2>&1; then
        echo "[Boot] Required packages not importable in venv (minor version mismatch). Rebuilding venv via setup_once.sh..."
        rm -rf /root/venv
        run_setup_once
    fi
fi

if [ ! -f "/root/venv/bin/python3" ]; then
    echo "ERROR: /root/venv/bin/python3 still missing after self-healing setup_once.sh!" >&2
    exit 1
fi

wait "${OLLAMA_EXTRACT_PID:-}" 2>/dev/null || true

# Verify restored version matches OLLAMA_VERSION; on mismatch or missing, reinstall and rewrite cache
INSTALLED_OLLAMA_VER=""
if command -v ollama >/dev/null 2>&1; then
    INSTALLED_OLLAMA_VER=$(ollama --version 2>&1 || echo "")
fi

if [ -z "$INSTALLED_OLLAMA_VER" ] || [[ "$INSTALLED_OLLAMA_VER" != *"$OLLAMA_VERSION"* ]]; then
    echo "[Boot] Warning: Ollama version mismatch or missing ('${INSTALLED_OLLAMA_VER}' vs required '${OLLAMA_VERSION}'). Reinstalling pinned version..."
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

echo "[Boot] Verified Ollama version: $(ollama --version 2>&1 || echo 'unknown')"

now=$(date +%s)
echo "[Boot] Phase [5/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "=== [6/8] Syncing Application Code to /root/app ==="
if [ -d "/workspace/app" ]; then
    cp -r /workspace/app/* /root/app/
fi

now=$(date +%s)
echo "[Boot] Phase [6/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "=== [7/8] Staging Models & Exporting Environment ==="
if [ -d "/workspace/models/ollama" ]; then
    echo "Linking Ollama models from /workspace to local SSD (/root/.ollama/models)..."
    mkdir -p /root/.ollama
    rm -rf /root/.ollama/models
    ln -sfn /workspace/models/ollama /root/.ollama/models
fi

if [ -d "/workspace/models/faster-whisper-large-v3" ]; then
    echo "Linking Faster-Whisper large-v3 from /workspace to local SSD (/root/models/faster-whisper-large-v3)..."
    mkdir -p /root/models
    rm -rf /root/models/faster-whisper-large-v3
    ln -sfn /workspace/models/faster-whisper-large-v3 /root/models/faster-whisper-large-v3
fi

export TMPDIR=/tmp
export HF_HOME=/tmp/hf_cache
export TORCH_HOME=/tmp/torch_cache
export PYTHONUNBUFFERED=1
export OLLAMA_MODELS=/root/.ollama/models
export OLLAMA_NUM_PARALLEL=2
export WHISPER_MODEL=/root/models/faster-whisper-large-v3

# Configure CUDA / cuDNN library paths
for nv_lib in /root/venv/lib/python*/site-packages/nvidia/*/lib; do
    if [ -d "$nv_lib" ]; then
        export LD_LIBRARY_PATH="${nv_lib}:${LD_LIBRARY_PATH:-}"
    fi
done
if [ -d "/usr/local/cuda/lib64" ]; then
    export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
fi

now=$(date +%s)
echo "[Boot] Phase [7/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "=== [8/8] Starting Services ==="
pkill -9 -f "ollama serve" 2>/dev/null || true
ollama serve &

# Fast readiness poll: check http://127.0.0.1:11434/api/tags every 0.2s (up to 25 iterations / 5s max)
for i in $(seq 1 25); do
    if curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
        break
    fi
    sleep 0.2
done

echo "[Boot] Ollama daemon running. Models will be initialized and pre-warmed by server."

# Background Auto-Migration: If volume only had uncompressed .tar files, migrate to .tar.zst
if [ ! -s "/workspace/venv_bundle.tar.zst" ] || [ ! -s "/workspace/ollama_bundle.tar.zst" ]; then
    (
        echo "[Migration] Background compressing bundles to zstd for instant future boots..."
        if [ ! -s "/workspace/ollama_bundle.tar.zst" ] && [ -d /usr/local/lib/ollama ]; then
            tar -I "zstd -1 -T0" -cf /tmp/ollama_bundle.tar.zst -C /usr/local bin/ollama lib/ollama 2>/dev/null || true
            if [ -s /tmp/ollama_bundle.tar.zst ]; then
                cp -f /tmp/ollama_bundle.tar.zst /workspace/ollama_bundle.tar.zst.tmp
                if zstd -t /workspace/ollama_bundle.tar.zst.tmp >/dev/null 2>&1; then
                    mv -f /workspace/ollama_bundle.tar.zst.tmp /workspace/ollama_bundle.tar.zst
                    rm -f /workspace/ollama_bundle.tar
                fi
                rm -f /workspace/ollama_bundle.tar.zst.tmp /tmp/ollama_bundle.tar.zst
            fi
        fi
        if [ ! -s "/workspace/venv_bundle.tar.zst" ] && [ -d /root/venv ]; then
            tar -I "zstd -1 -T0" -cf /tmp/venv_bundle.tar.zst -C /root venv 2>/dev/null || true
            if [ -s /tmp/venv_bundle.tar.zst ]; then
                cp -f /tmp/venv_bundle.tar.zst /workspace/venv_bundle.tar.zst.tmp
                if zstd -t /workspace/venv_bundle.tar.zst.tmp >/dev/null 2>&1; then
                    mv -f /workspace/venv_bundle.tar.zst.tmp /workspace/venv_bundle.tar.zst
                    rm -f /workspace/venv_bundle.tar
                fi
                rm -f /workspace/venv_bundle.tar.zst.tmp /tmp/venv_bundle.tar.zst
            fi
        fi
        echo "[Migration] Completed zstd bundle migration on /workspace."
    ) >/dev/null 2>&1 &
fi

# Clear any setup failure counter so running server starts with a clean slate
rm -f /workspace/.setup_failures

now=$(date +%s)
echo "[Boot] Phase [8/8] completed in $(( now - LAST_PHASE_TIME ))s (Total: $(( now - BOOT_START_TIME ))s)"
LAST_PHASE_TIME=$now

echo "Launching Live Translation Server on port 8000..."
cd /root/app
exec /root/venv/bin/python3 runpod_server.py
