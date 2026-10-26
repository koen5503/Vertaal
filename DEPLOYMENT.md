# RunPod Mini-Pod + Global Volume Deployment Guide

This guide documents the architecture, configuration, and operational workflow for running the Ondertitels live transcription & translation backend on RunPod using the **Mini-Pod + Global Volume** pattern.

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
   - Pod boot-up time is slashed to **< 15 seconds**.

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

## 3. Pod Configuration Specifications

When creating the RunPod Template or deploying a Pod on the RunPod Console:

| Setting | Recommended Value | Notes |
| :--- | :--- | :--- |
| **Base Docker Image** | `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` | Pre-cached globally across RunPod GPU nodes; zero container pull latency. |
| **Container Disk** | `20 GB` | Local NVMe space for `/root/venv` (~5 GB unpacked) and `/tmp` buffers. |
| **Volume Disk** | Attach your **Global Volume** | Mounted at `/workspace` (e.g. 50 GB capacity). |
| **Expose HTTP Ports** | `8000` | Port used by FastAPI & WebSocket server. |
| **Docker Start Command** | `bash -c "bash /workspace/boot.sh"` | Starts Ollama, unpacks venv, and launches `runpod_server.py`. |

---

## 4. One-Time Setup Procedure (`setup_once.sh`)

Follow this procedure **only once** when initializing a fresh Global Volume:

1. Deploy a temporary Pod with your Global Volume mounted at `/workspace`.
2. Copy `setup_once.sh`, `boot.sh`, and `runpod_server.py` to `/workspace/`:
   ```bash
   scp setup_once.sh boot.sh runpod_server.py <pod-ip>:/workspace/
   ```
   *(Alternatively, clone the repository directly into `/workspace`)*.
3. Open a Web Terminal or SSH session on the Pod and run:
   ```bash
   bash /workspace/setup_once.sh
   ```
4. `setup_once.sh` will automatically:
   - Install system dependencies (`zstd`, `curl`, `procps`, `ffmpeg`, `libsndfile1`).
   - Install the Ollama engine.
   - Build `/root/venv` with all necessary packages (`faster-whisper`, `ollama`, `fastapi`, `uvicorn`, etc.).
   - Download Faster-Whisper `large-v3` flat to `/workspace/models/faster-whisper-large-v3` without symlinks.
   - Pull `aya-expanse:8b` directly into `/workspace/models/ollama`.
   - Archive `/root/venv` to `/workspace/venv_bundle.tar`.
   - Copy `runpod_server.py` into `/workspace/app/runpod_server.py`.
5. Terminate the setup pod once the script outputs: `=== One-time setup completed successfully! ===`.

---

## 5. Daily Operation & Fast Boot (< 15 seconds)

For normal live subtitle sessions:
1. Launch any high-bandwidth GPU pod (RTX 4090, RTX 6000 Ada, RTX 4080, RTX 3090) with the Global Volume attached.
2. Set the Container Start Command:
   ```bash
   bash -c "bash /workspace/boot.sh"
   ```
3. `boot.sh` will:
   - Unpack `/workspace/venv_bundle.tar` into local NVMe `/root/venv` in ~5 seconds.
   - Sync `/workspace/app/*` into `/root/app/`.
   - Start Ollama against `/workspace/models/ollama`.
   - Launch `runpod_server.py` with instant local load of Faster-Whisper and Ollama.
4. Total ready-to-serve time is **under 15 seconds**.

---

## 6. Zero-Rebuild Code Updates

When making changes to `runpod_server.py`:
- You do **not** need to rebuild the venv, download models, or recreate templates.
- Simply update `/workspace/app/runpod_server.py`:
  ```bash
  cp runpod_server.py /workspace/app/runpod_server.py
  ```
- Any newly started pod will immediately sync and run the updated server code.
