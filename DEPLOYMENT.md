# RunPod Mini-Pod + Global Volume Deployment Guide

This guide documents the architecture, configuration, and **100% automated self-initializing deployment pipeline** for running the Ondertitels live transcription & translation backend on RunPod using the **Mini-Pod + Global Volume** pattern.

---

## 1. Architecture Overview

### Why Mini-Pod + Global Volume?
Traditional cloud GPU containers often rebuild environments or pull large AI models (such as Faster-Whisper `large-v3` ~3 GB and Ollama `aya-expanse:8b` ~5 GB) on every pod spin-up, incurring 5–10 minutes of latency and excess network costs.

The **Mini-Pod + Global Volume** architecture separates persistent state from volatile execution:
1. **Persistent Global Network Volume (`/workspace`)**:
   - Backed by high-capacity S3 object storage mounted via FUSE.
   - Houses pre-downloaded AI models (`/workspace/models/`) and a pre-bundled Python virtual environment (`/workspace/venv_bundle.tar`).
   - Persists across pod restarts, terminations, and GPU model switches.
2. **Volatile Container NVMe SSD (`/root` & `/tmp`)**:
   - The Python virtual environment is extracted to `/root/venv` on the host's high-speed local NVMe disk.
   - All caches and temporary files are strictly bound to `/tmp` (`TMPDIR=/tmp`, `HF_HOME=/tmp/hf_cache`, `TORCH_HOME=/tmp/torch_cache`).
   - Pod boot-up time is slashed to **< 15 seconds** for daily operations.

```
+-----------------------------------------------------------------------+
| RunPod Host (Ada Lovelace / Ampere GPU: RTX 4090 / 6000 Ada / 3090)  |
|                                                                       |
|  +---------------------------+       +-----------------------------+  |
|  | Volatile Local NVMe SSD   |       | Persistent Global Volume    |  |
|  | (/root, /tmp)             |       | (/workspace - FUSE S3)      |  |
|  +---------------------------+       +-----------------------------+  |
|  | - /root/venv              | <===  | - venv_bundle.tar           |  |
|  |   (extracted tarball)     |  tar  | - models/                   |  |
|  | - /root/app/              | <===  |   * faster-whisper-large-v3 |  |
|  |   (runpod_server.py)      |  cp   |   * ollama/aya-expanse:8b   |  |
|  | - /tmp/hf_cache           |       | - app/runpod_server.py      |  |
|  | - /tmp/torch_cache        |       | - boot.sh                   |  |
|  |                           |       | - setup_once.sh             |  |
|  +---------------------------+       +-----------------------------+  |
+-----------------------------------------------------------------------+
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
   - The virtual environment is archived into a single tar archive (`/workspace/venv_bundle.tar`) and unpacked to local NVMe `/root/venv` on boot.
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
| **Docker Start Command** | `bash -c "[ -f /workspace/boot.sh ] && bash /workspace/boot.sh \|\| (curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh)"` | Self-seeding bootstrap command. |

---

## 4. 100% Automated Deployment Lifecycle

**Zero manual SSH or terminal commands are required.** The entire lifecycle is managed automatically by `app.py` and `core/stt/runpod_engine.py`:

### First Run on Clean / Empty Global Volume (~3–5 minutes):
1. When `app.py` starts in `runpod` mode, it auto-rents an available GPU pod with your Global Volume mounted.
2. The pod starts with the self-seeding Docker command.
3. If `/workspace/boot.sh` is missing, `boot.sh` downloads itself from GitHub into `/workspace/boot.sh`.
4. `boot.sh` detects that `/workspace` is brand new (missing `venv_bundle.tar` or models).
5. It automatically triggers `setup_once.sh`:
   - Installs OS tools (`ffmpeg`, `libsndfile1`, `curl`, `procps`, `zstd`).
   - Installs Ollama.
   - Builds `/root/venv` and installs packages (`fastapi`, `uvicorn`, `faster-whisper`, etc.).
   - Downloads Faster-Whisper `large-v3` flat to `/workspace/models/faster-whisper-large-v3`.
   - Pulls `aya-expanse:8b` directly to `/workspace/models/ollama`.
   - Archives `/root/venv` into `/workspace/venv_bundle.tar`.
   - Copies `runpod_server.py` to `/workspace/app/runpod_server.py`.
6. `condense_runpod_logs()` detects this phase and reports:
   `"Automatische eerste initialisatie van Global Volume (duurt ~3-5m)..."` directly in the Telemetry dashboard.
7. Once finished, the server begins streaming immediately.

### Subsequent Daily Runs (< 15 seconds):
1. `app.py` discovers or rents a GPU pod.
2. `boot.sh` executes immediately from `/workspace/boot.sh`.
3. It detects that `venv_bundle.tar` and models already exist:
   - Unpacks `/workspace/venv_bundle.tar` to `/root/venv` in ~5 seconds.
   - Syncs `/workspace/app/*` to `/root/app/`.
   - Starts Ollama daemon.
   - Launches `runpod_server.py` on port 8000.
4. Total ready-to-serve time is **under 15 seconds**.

---

## 5. Zero-Rebuild Code Updates

When making changes to `runpod_server.py`:
- You do **not** need to rebuild the venv, download models, or recreate templates.
- Update `/workspace/app/runpod_server.py` directly:
  ```bash
  cp runpod_server.py /workspace/app/runpod_server.py
  ```
- Or commit to GitHub: upon container restart or deployment, `boot.sh` automatically ensures the latest code is synced to `/root/app/`.
