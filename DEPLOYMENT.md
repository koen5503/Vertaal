# RunPod Mini-Pod + Global Volume Deployment Guide

This guide documents the architecture, configuration, and **100% automated self-initializing deployment pipeline** for running the Ondertitels live transcription & translation backend on RunPod using the **Mini-Pod + Global Volume** pattern.

---

## 1. Architecture Overview

### Why Mini-Pod + Global Volume?
Traditional cloud GPU containers often rebuild environments or pull large AI models (such as Faster-Whisper `large-v3` ~3 GB and Ollama `aya-expanse:8b` ~5 GB) on every pod spin-up, incurring 5–10 minutes of latency and excess network costs.

The **Mini-Pod + Global Volume** architecture separates persistent state from volatile execution:
1. **Persistent Global Network Volume (`/workspace`)**:
   - Backed by high-capacity S3 object storage mounted via FUSE.
   - Houses pre-downloaded AI models (`/workspace/models/`) and zstd-compressed archives (`/workspace/venv_bundle.tar.zst` and `/workspace/ollama_bundle.tar.zst`, with legacy `.tar` fallback), tracked by `/workspace/.bundle_version`.
   - Persists across pod restarts, terminations, and GPU model switches.
2. **Volatile Container NVMe SSD (`/root` & `/tmp`)**:
   - The Python virtual environment is extracted to `/root/venv` on the host's high-speed local NVMe disk.
   - All caches and temporary files are strictly bound to `/tmp` (`TMPDIR=/tmp`, `HF_HOME=/tmp/hf_cache`, `TORCH_HOME=/tmp/torch_cache`).
   - Container boot and virtual environment extraction take **< 15 seconds**, while full operational readiness takes **~40–55 seconds** (including loading 8.3 GB of Faster-Whisper large-v3 and Ollama weights into GPU VRAM).

```
+----------------------------------------------------------------------------------+
| RunPod Host (Ada Lovelace / Ampere GPU: RTX 4090 / 6000 Ada / 3090 / A5000)      |
|                                                                                  |
|  +------------------------------+       +-------------------------------------+  |
|  | Volatile Local NVMe SSD      |       | Persistent Global Volume            |  |
|  | (/root, /tmp)                |       | (/workspace - FUSE S3)              |  |
|  +------------------------------+       +-------------------------------------+  |
|  | - /root/venv                 | <===  | - venv_bundle.tar.zst               |  |
|  |   (extracted tarball)        |  tar  |   (fallback: venv_bundle.tar)       |  |
|  | - /root/app/                 | <===  | - ollama_bundle.tar.zst             |  |
|  |   * runpod_server.py         |  cp   | - .bundle_version                   |  |
|  |   * core/                    |       | - models/                           |  |
|  | - /tmp/hf_cache              |       |   * faster-whisper-large-v3         |  |
|  | - /tmp/torch_cache           |       |   * ollama/aya-expanse:8b           |  |
|  |                              |       | - app/runpod_server.py              |  |
|  |                              |       | - app/core/                         |  |
|  |                              |       | - boot.sh                           |  |
|  |                              |       | - setup_once.sh                     |  |
|  +------------------------------+       +-------------------------------------+  |
+----------------------------------------------------------------------------------+
```

---

## 2. Crucial Storage & Technical Constraints

The RunPod Global Volume (`/workspace`) is an S3-backed Object Storage system mounted through FUSE. Violating the following constraints will cause container crashes or I/O hangs:

1. **NO Symlinks or Hardlinks**:
   - Hugging Face's default hub cache relies heavily on symlinks (`snapshots/` pointing to `blobs/`), which **fail and crash** on FUSE S3 mounts.
   - Faster-Whisper models must be stored completely flat in `/workspace/models/faster-whisper-large-v3` with `--local-dir-use-symlinks False`.
2. **NO `chmod +x` Execution**:
   - FUSE mounts may not support executable bit persistence (`chmod +x`).
   - Always invoke scripts explicitly via bash: `bash /workspace/boot.sh` or `bash /workspace/setup_once.sh`.
3. **NO Millions of Small Files**:
   - Running an active Python virtual environment directly on a FUSE mount introduces extreme filesystem latency and kernel locks.
   - The virtual environment is archived into `/workspace/venv_bundle.tar.zst` (with uncompressed `.tar` fallback) and unpacked to local NVMe `/root/venv` on boot.
4. **Strict Isolation of Volatile Caches**:
   - `TMPDIR=/tmp`, `HF_HOME=/tmp/hf_cache`, and `TORCH_HOME=/tmp/torch_cache` must always point to `/tmp`, never `/workspace`.

---

## 3. Pod Configuration Specifications (RunPod Template)

Configure your RunPod Template once on the RunPod Console:

| Setting | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Base Docker Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Pre-cached globally across RunPod GPU nodes; zero container pull latency. |
| **Container Disk** | `20 GB` | Local NVMe space for `/root/venv` (~5 GB unpacked) and `/tmp` buffers. |
| **Volume Disk** | Attach your **Global Volume** | Mounted at `/workspace` (e.g. 50 GB capacity). |
| **Expose HTTP Ports** | `8000` | Port used by FastAPI & WebSocket server. |
| **Docker Start Command** | `bash -c "[ -f /workspace/boot.sh ] && bash /workspace/boot.sh \|\| (curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/\${RUNPOD_REPO_BRANCH:-feature/local-mac-ai-pipeline}/boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh \|\| curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/main/boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh)"` | Self-seeding bootstrap command. |

---

## 4. 100% Automated Deployment Lifecycle

**Zero manual SSH or terminal commands are required.** The entire lifecycle is managed automatically by `app.py` and `core/stt/runpod_engine.py`:

### First Run on Clean / Empty Global Volume (~3–5 minutes):
1. When `app.py` starts in `runpod` mode, it auto-rents an available GPU pod with your Global Volume mounted.
2. The pod starts with the self-seeding Docker command.
3. If `/workspace/boot.sh` is missing, `boot.sh` downloads itself from GitHub into `/workspace/boot.sh`.
4. `boot.sh` detects that `/workspace` is brand new (missing `venv_bundle.tar.zst` / `venv_bundle.tar` or models).
5. It automatically triggers `setup_once.sh`:
   - Installs OS tools (`ffmpeg`, `libsndfile1`, `curl`, `procps`, `zstd`).
   - Installs Ollama and creates `/workspace/ollama_bundle.tar.zst` atomically (`.tmp` + `zstd -t` + `mv -f`).
   - Builds `/root/venv` and installs packages (`fastapi`, `uvicorn`, `faster-whisper`, etc.).
   - Downloads Faster-Whisper `large-v3` flat to `/workspace/models/faster-whisper-large-v3`.
   - Pulls `aya-expanse:8b` directly to `/workspace/models/ollama`.
   - Archives `/root/venv` into `/workspace/venv_bundle.tar.zst` atomically (`.tmp` + `zstd -t` + `mv -f`), retaining legacy `venv_bundle.tar` compatibility.
   - Writes `/workspace/.bundle_version` manifest for version pinning and upgrade detection.
   - Stages `runpod_server.py`, `boot.sh`, and `app/core/` packages into `/workspace/app/core/`.
6. `condense_runpod_logs()` detects this phase and reports:
   `"Automatische eerste initialisatie van Global Volume (duurt ~3-5m)..."` directly in the Telemetry dashboard.
7. Once finished, the server begins streaming immediately.

### Subsequent Daily Runs (< 15s boot, ~40–55s full readiness):
1. `app.py` discovers or rents a GPU pod.
2. `boot.sh` executes immediately from `/workspace/boot.sh`.
3. It detects that `venv_bundle.tar.zst` (or `venv_bundle.tar`) and models already exist:
   - Unpacks `/workspace/venv_bundle.tar.zst` (with legacy `/workspace/venv_bundle.tar` fallback) to `/root/venv` in ~5 seconds.
   - Syncs `/workspace/app/*` (including `app/core/`) to `/root/app/`.
   - Starts Ollama daemon and pre-warms translation models.
   - Launches `runpod_server.py` on port 8000.
4. Container boot and virtual environment extraction complete in **< 15 seconds**, reaching full operational readiness in **~40–55 seconds** (including loading 8.3 GB of Faster-Whisper large-v3 and Ollama weights into GPU VRAM).

---

## 5. Zero-Rebuild Code Updates

When making changes to `runpod_server.py`:
- You do **not** need to rebuild the venv, download models, or recreate templates.
- Update `/workspace/app/runpod_server.py` directly:
  ```bash
  cp runpod_server.py /workspace/app/runpod_server.py
  ```
- Or commit to GitHub: upon container restart or deployment, `boot.sh` automatically ensures the latest code is synced to `/root/app/`.

---

## 6. Upgrading Pod Dependencies & Rebuilding Environment

When updating Python dependencies or the Ollama runtime version:

1. **Edit dependency pins**:
   - Update package versions in `requirements-pod.txt` using exact `==` pins.
   - If upgrading Ollama, update `OLLAMA_VERSION` (e.g. `"0.5.7"`).

2. **Compute new bundle version hash**:
   ```bash
   python3 -c 'import hashlib; c=open("requirements-pod.txt","rb").read()+b"\nOLLAMA_VERSION=0.5.7"; print("2026.10.2+" + hashlib.sha256(c).hexdigest()[:8])'
   ```

3. **Update `BUNDLE_VERSION` in boot scripts**:
   - Set `BUNDLE_VERSION="2026.10.2+<sha8>"` identically in both `boot.sh` and `setup_once.sh`.
   - Update `OLLAMA_VERSION="<version>"` identically in both scripts.

4. **Run Pre-Flight Verification Gate**:
   ```bash
   python3 -m unittest discover -s tests -p "test_*.py" -v
   ```
   All 170+ unit and simulation tests must pass 100% before committing or deploying.

5. **Deploy & Live E2E Verification**:
   - On the next pod start, `boot.sh` compares the volume manifest `/workspace/.bundle_version` with `BUNDLE_VERSION`.
   - On mismatch, `boot.sh` automatically runs `setup_once.sh`:
     - Synchronizes `requirements-pod.txt` to `/workspace/app/requirements-pod.txt`.
     - Rebuilds `/root/venv` with `/root/venv/bin/pip install --no-cache-dir -r /workspace/app/requirements-pod.txt`.
     - Verifies `ollama --version` matches `OLLAMA_VERSION`, reinstalling if necessary.
     - Atomically rewrites `/workspace/venv_bundle.tar.zst` and writes the new `BUNDLE_VERSION` to `/workspace/.bundle_version`.
   - Verify live audio transcription and translation connectivity over the WebSocket.
