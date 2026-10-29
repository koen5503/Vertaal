# Live Subtitles & Translation ("Ondertitels")

Real-time speech-to-text subtitling with simultaneous multi-lingual translation and chained back-translations. The application is built for live events, church services, and presentations, featuring a unified FastAPI backend that flexibly supports three execution backends: **Local Apple Silicon (macOS MLX)**, **Google Cloud Speech-to-Text & Translation**, and **Remote NVIDIA GPU Acceleration (RunPod)**.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![Docker-Free](https://img.shields.io/badge/Client-100%25%20Docker--Free-success)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![Tests](https://img.shields.io/badge/Tests-165%20Passing-brightgreen)
![License](https://img.shields.io/badge/License-Private-red)

---

## 🌟 Key Features

- **3 Flexible Pipeline Modes** — Choose between 100% private on-device processing, ultra-low latency Google Cloud streaming, or high-throughput remote NVIDIA GPU acceleration on RunPod.
- **100% Docker-Free Client** — Runs natively on macOS and Windows with zero local container overhead. No Docker Desktop, local daemon, or WSL2 required on client machines.
- **5-Column Subtitle Display** — Live source speech, two direct target translations, and two chained back-translations for real-time verification and multi-lingual reach.
- **Full-Segment Revision Protocol** — Cumulative audio chunks replace interim versions directly in-place without concatenation lag or word duplication; interim translations display in cursive/italic styling until finalized.
- **Resilient Anti-Hallucination Filtering** — Dynamic sliding-window context memory cue with automated prefix-overlap trimming, repetition loop suppression, and breath/cough interim suppression without dropping subsequent speech.
- **Advanced Audio Processing & VAD** — WebRTC Voice Activity Detection (exact 10ms, 20ms, or 30ms frames), gain normalization, automatic audio format resampling (48kHz/44.1kHz downmix to 16kHz mono), and RMS volume metering.
- **Multiple Audio Input Sources** — CoreAudio / ASIO / WASAPI microphones, VoiceMeeter virtual audio routing (BUS B1), network audio streaming via VBAN (UDP port 6980), or offline WAV file playback.
- **Ephemeral RunPod Mini-Pod Architecture** — On-demand ephemeral pod provisioning with persistent Global Volume storage (`/workspace`), automated self-seeding, fast boot (virtual environment unpack & script bootstrap in < 15s; full model VRAM readiness in ~40–55s) using standard RunPod PyTorch base images (zero custom Docker Hub images needed), automatic zombie pod cleanup, and idle auto-stop.
- **Live Telemetry & Health Monitoring** — Dedicated real-time dashboard displaying phased container boot stages `[1/8]`–`[8/8]`, GPU hardware stats, STT/translation latencies, and WebSocket connection health.
- **Dynamic Sliding Session Window** — Automatic activity-based session extensions bounded by a hard safety ceiling to prevent runaway cloud GPU costs.

---

## 🏗 Architecture & Execution Modes

The application captures audio through PyAudio or UDP network packets, preprocesses the signal into 16kHz 16-bit mono PCM chunks, applies WebRTC Voice Activity Detection, and routes segments to one of three configurable STT engines:

```text
 ┌────────────────────────────────────────────────────────┐
 │                      Audio Sources                     │
 │  Microphone (CoreAudio/WASAPI) │ VoiceMeeter B1 │ VBAN │
 └───────────────────────────┬────────────────────────────┘
                             │
                             ▼
 ┌────────────────────────────────────────────────────────┐
 │           Audio Preprocessing & WebRTC VAD             │
 │  48k/44.1k -> 16kHz Mono PCM │ 30ms Frames │ Gain Meter│
 └───────────────────────────┬────────────────────────────┘
                             │
            ┌────────────────┴────────────────┐
            │ PIPELINE_MODE Selection in .env │
            └────────────────┬────────────────┘
                             │
     ┌───────────────────────┼───────────────────────┐
     ▼                       ▼                       ▼
┌──────────────┐      ┌──────────────┐      ┌─────────────────┐
│ Mode 1:      │      │ Mode 2:      │      │ Mode 3:         │
│ Local macOS  │      │ Google Cloud │      │ RunPod Remote   │
│ Apple Silicon│      │ Streaming    │      │ GPU (WebSocket) │
│ (MLX-Whisper)│      │ (Cloud STT)  │      │ (Faster-Whisper)│
└──────┬───────┘      └──────┬───────┘      └────────┬────────┘
       │                     │                       │
       ▼                     ▼                       ▼
 ┌───────────┐         ┌───────────┐           ┌───────────┐
 │ Ollama LLM│         │ Google    │           │ Ollama on │
 │ Translate │         │ Translate │           │ RunPod /  │
 │ (Local)   │         │ Cloud API │           │ Local     │
 └─────┬─────┘         └─────┬─────┘           └─────┬─────┘
       │                     │                       │
       └─────────────────────┼───────────────────────┘
                             │
                             ▼
       ┌───────────────────────────────────────────┐
       │     FastAPI WebSocket Real-Time Broadcast │
       └─────────────────────┬─────────────────────┘
                             │
                             ▼
  ┌─────────────────────────────────────────────────────────┐
  │                 5-Column Live Display                   │
  │  Col 1: Source Speech (e.g. Dutch)                      │
  │  Col 2: Direct Translation 1 (e.g. English)             │
  │  Col 3: Direct Translation 2 (e.g. Russian)             │
  │  Col 4: Chained Back-Translation 1 (Col 2 -> e.g. German│
  │  Col 5: Chained Back-Translation 2 (Col 3 -> e.g. Dutch)│
  └─────────────────────────────────────────────────────────┘
```

> [!NOTE]
> **100% Docker-Free Client**: The host machine running Ondertitels (macOS or Windows) runs pure native Python without requiring Docker Desktop, container daemons, or WSL2 virtualization. When remote GPU acceleration is enabled, the client communicates over secure WebSocket with standard PyTorch cloud pods.

### The 3 Execution Backends

1. **Local Apple Silicon (`PIPELINE_MODE=local`)**
   - **STT**: `mlx-whisper` running natively on macOS Metal GPU and Apple Neural Engine.
   - **Translation**: Local LLM served via `Ollama` (e.g., `aya-expanse:8b` or `qwen2.5:3b`).
   - **Characteristics**: 100% offline, zero cloud API costs, complete data privacy.

2. **Google Cloud (`PIPELINE_MODE=cloud`)**
   - **STT**: Google Cloud Speech-to-Text v1 streaming client with real-time stability tracking.
   - **Translation**: Google Cloud Translation API (v3/v2).
   - **Characteristics**: Lowest transcription latency for live spoken words; ideal for high-stakes broadcast environments with reliable internet.

3. **Remote NVIDIA GPU on RunPod (`PIPELINE_MODE=runpod`)**
   - **Architecture**: **Mini-Pod + Global Volume (`/workspace`)** pattern running on remote NVIDIA GPUs (RTX 3090, RTX 4090, A4000, A5000).
   - **Zero Custom Docker Hub Images**: Deploys using standard RunPod PyTorch base images (`runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`). No Docker Hub repository or custom image builds needed.
   - **Automated Self-Seeding & Fast Boot**: On a fresh volume, `boot.sh` automatically seeds the environment via `setup_once.sh` in <3–5 minutes. Subsequent pod starts unpack the environment from `/workspace` in **< 15 seconds**, reaching full operational readiness in **~40–55 seconds** (including loading 8.3 GB of Faster-Whisper large-v3 and Ollama weights into GPU VRAM).
   - **Lifecycle Management**: On-demand ephemeral pod provisioning with persistent storage on the RunPod Global Volume, automatic zombie pod cleanup, and idle auto-stop.
   - **STT & Translation**: `faster-whisper` (`large-v3`) with local NVMe SSD staging for high I/O throughput + `Ollama` (`aya-expanse:8b`) with GPU VRAM pre-warming.
   - **Communication**: Bidirectional streaming over secure WebSocket (`runpod_server.py`).

---

## 📝 Full-Segment Revision & Subtitle Quality

Ondertitels uses a **Full-Segment Revision** protocol designed to maximize subtitle readability and translation quality during live continuous speech:

- **In-Place Segment Replacement**: Rather than concatenating fragmented words as audio streams in, the audio engine sends cumulative audio chunks representing the full utterance to date. The STT engine transcribes the complete chunk with full acoustic and grammatical context, replacing the interim version directly in the UI. This eliminates word stutter, repeated prefixes, and sentence splicing artifacts.
- **Visual Status Differentiation**:
  - **Interim Subtitles**: Rendered in *italic / cursive* styling with subtle dashed borders while the speaker is mid-sentence.
  - **Finalized Subtitles**: Rendered in regular font weight with solid borders once a natural pause occurs (or when forced by silence threshold). Finalized segments are protected against regression.
- **Non-Destructive Hallucination Suppression**: Whisper models can occasionally hallucinate phantom phrases or subtitle credits (e.g. `***`, `Ondertiteling: ...`, coughs, mic pops) during quiet or breathing intervals. The engine suppresses these hallucination chunks without permanently blocking the segment ID. When genuine speech continues after the pause, the segment is immediately unsuppressed and translations flow smoothly without losing subsequent words.

---

## ⏱ Session Window Management & Safety Ceiling

To prevent runaway cloud compute billing and abandoned GPU instances, the backend integrates an automated **Dynamic Sliding Session Window**:

- **Initial Window (`INITIAL_SESSION_WINDOW_MINUTES=15`)**: When a session starts or connects to RunPod, the timer initializes with a 15-minute operational window.
- **Sliding Activity Extension (`EXTEND_WINDOW_MINUTES=10`)**: Every time speech activity or STT transcription is detected, the sliding window automatically extends by 10 minutes from the current timestamp.
- **Hard Safety Ceiling (`MAX_SESSION_CEILING_MINUTES=90`)**: Total continuous runtime is capped by a strict ceiling (default: 90 minutes). The sliding window cannot extend beyond this point regardless of ongoing speech, protecting against forgotten overnight sessions.
- **Silence Heartbeat Probes (`SILENCE_HEARTBEAT_SEC=60`)**: During extended pauses or sermon transitions, the client dispatches periodic heartbeat probes over the WebSocket to maintain connection liveness and monitor container telemetry.
- **Auto-Stop Enforcement**: When the window expires or the ceiling is reached, the pipeline automatically pauses audio capture and terminates/stops the ephemeral RunPod pod.

---

## 📊 5-Column Display & Chained Translations

The frontend and broadcast pipeline manage 5 concurrent text streams per speech segment:

| Column | Role | Data Flow | Typical Setting | Purpose |
|---|---|---|---|---|
| **Col 1** | Source Transcription | Audio → STT Engine | Dutch (`nl-NL`) | Verbatim transcript of speaker |
| **Col 2** | Target Translation 1 | Col 1 → Translator | English (`en`) | Primary foreign language display |
| **Col 3** | Target Translation 2 | Col 1 → Translator | Russian (`ru`) | Secondary foreign language display |
| **Col 4** | Chained Translation 1 | Col 2 → Translator | German (`de`) or Dutch (`nl`) | Back-translation verification or 3rd language |
| **Col 5** | Chained Translation 2 | Col 3 → Translator | Dutch (`nl-NL`) | Back-translation verification to source language |

Columns 4 and 5 can be toggled in real time via `.env` (`ENABLE_TRANS_3`, `ENABLE_TRANS_4`) or via the web admin dashboard. Chained translations enable bilingual operators to instantly spot inaccuracies in intermediate translations and provide extra languages with zero manual overhead.

---

## 🚀 Setup & Execution Guides

### 🍎 Backend 1: Local macOS Setup (Apple Silicon)

1. **Install Prerequisites**:
   ```bash
   brew install portaudio ollama
   ```
2. **Install Python Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
3. **Pull Translation Model and Start Ollama**:
   ```bash
   ollama pull aya-expanse:8b
   ollama serve
   ```
4. **Configure `.env`**:
   ```env
   PIPELINE_MODE=local
   WHISPER_MODEL=mlx-community/whisper-small-mlx
   OLLAMA_MODEL=aya-expanse:8b
   OLLAMA_URL=http://localhost:11434
   ```
5. **Launch Application**:
   ```bash
   python3 app.py
   ```

---

### ☁️ Backend 2: Google Cloud Setup

1. **Install Python Dependencies**:
   ```bash
   pip3 install -r requirements.txt
   ```
2. **Place Service Account Key**:
   Save your Google Cloud service account key JSON file in the project directory.
3. **Configure `.env`**:
   ```env
   PIPELINE_MODE=cloud
   GOOGLE_APPLICATION_CREDENTIALS=your-service-account-key.json
   GOOGLE_PROJECT_ID=your-gcp-project-id
   STABILITY_THRESHOLD=0.8
   MIN_WORDS_FOR_STABILITY=5
   ```
4. **Launch Application**:
   ```bash
   python3 app.py
   ```

---

### ⚡ Backend 3: Windows / macOS & RunPod Cloud STT

Optimized for client machines (Windows PC connected to mixers like Behringer X-32 via VoiceMeeter, or macOS) connecting to on-demand NVIDIA GPU pods in the cloud. **Zero Docker Desktop or local container runtimes required on the client machine.**

1. **Prerequisites**:
   - Python 3.10+ (or use compiled `app.exe` on Windows)
   - On Windows: [VoiceMeeter](https://voicemeeter.com/) (Standard, Banana, or Potato)
   - [RunPod](https://www.runpod.io/) account with API key and a Network Global Volume (`/workspace`)
2. **Windows Firewall Configuration (PowerShell as Administrator)**:
   ```powershell
   New-NetFirewallRule -DisplayName "RunPod Outbound" -Direction Outbound -Action Allow -Protocol TCP -RemotePort 443,30000-65535
   New-NetFirewallRule -DisplayName "Ondertitels Viewer" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 80,8000
   ```
3. **Audio Routing**:
   Route your mixer or microphone input in VoiceMeeter to **BUS B1** (or select your local microphone device on macOS).
4. **Configure `.env`**:
   ```env
   PIPELINE_MODE=runpod
   RUNPOD_API_KEY=your_runpod_api_key
   RUNPOD_WHISPER_MODEL=large-v3
   RUNPOD_GPU_ID=NVIDIA RTX A5000
   AUTO_DEPLOY_RUNPOD=true
   MAX_RUN_TIME_SEC=5400
   INITIAL_SESSION_WINDOW_MINUTES=15
   EXTEND_WINDOW_MINUTES=10
   MAX_SESSION_CEILING_MINUTES=90
   SILENCE_HEARTBEAT_SEC=60
   ```
5. **Launch Application**:
   ```bash
   python app.py
   ```

---

## 🛠 Configuration Reference (`.env`)

Copy `.env.example` to `.env` to configure the application:

| Variable | Description | Mode | Default |
|---|---|---|---|
| `PIPELINE_MODE` | Active pipeline backend (`local`, `cloud`, or `runpod`) | All | `local` |
| `SOURCE_LANG` | BCP-47 language tag for spoken audio input | All | `nl-NL` |
| `TARGET_LANG_1` | First target translation language | All | `en` |
| `TARGET_LANG_2` | Second target translation language | All | `ru` |
| `TARGET_LANG_3` | Third language (chained from Target 1) | All | `de` |
| `TARGET_LANG_4` | Fourth language (chained from Target 2) | All | `nl-NL` |
| `ENABLE_TRANS_3` | Enable 4th column chained translation (`true`/`false`) | All | `false` |
| `ENABLE_TRANS_4` | Enable 5th column chained translation (`true`/`false`) | All | `false` |
| `TRANSLATE_INTERVAL_SEC` | Throttle interval between interim translations (seconds) | All | `10` |
| `INITIAL_SESSION_WINDOW_MINUTES` | Initial sliding session duration before auto-pausing (minutes) | All | `15` |
| `EXTEND_WINDOW_MINUTES` | Sliding window extension duration upon speech activity (minutes) | All | `10` |
| `MAX_SESSION_CEILING_MINUTES` | Strict safety ceiling capping total continuous session length (minutes) | All | `90` |
| `SILENCE_HEARTBEAT_SEC` | Probe interval sent across WebSocket during prolonged silence (seconds) | RunPod | `60` |
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud Service Account JSON file | Cloud | `""` |
| `GOOGLE_PROJECT_ID` | Google Cloud project identifier | Cloud | `""` |
| `STABILITY_THRESHOLD` | Minimum Google STT stability metric to trigger translation | Cloud | `0.8` |
| `MIN_WORDS_FOR_STABILITY` | Minimum words in transcript before interim translation | Cloud | `5` |
| `WHISPER_MODEL` | Hugging Face model identifier for MLX-Whisper | Local | `mlx-community/whisper-small-mlx` |
| `OLLAMA_MODEL` | Ollama model tag for LLM translation | Local / RunPod | `aya-expanse:8b` |
| `OLLAMA_URL` | Ollama service endpoint | Local / RunPod | `http://localhost:11434` |
| `RUNPOD_API_KEY` | RunPod API key (REST v2 / GraphQL) | RunPod | `""` |
| `RUNPOD_WSS_URL` | Manual WebSocket URL override (skips pod discovery) | RunPod | `""` |
| `RUNPOD_TEMPLATE_ID` | Template ID configured with standard PyTorch base image | RunPod | `""` |
| `RUNPOD_VOLUME_ID` | Network Global Volume mounted to `/workspace` | RunPod | `""` |
| `RUNPOD_GPU_ID` | Preferred GPU type for pod rental | RunPod | `NVIDIA RTX A5000` |
| `RUNPOD_DATA_CENTER_ID` | Optional datacenter code (e.g. `EU-RO-1`) | RunPod | `""` |
| `RUNPOD_WHISPER_MODEL` | Whisper model loaded on remote pod (`small`, `large-v3`) | RunPod | `large-v3` |
| `AUTO_DEPLOY_RUNPOD` | Automatically spin up ephemeral pod on startup | RunPod | `false` |
| `MAX_RUN_TIME_SEC` | Idle timeout for RunPod server self-termination (seconds) | RunPod | `5400` |
| `DEFAULT_DEVICE_NAME` | Audio input device auto-selected on boot | All | `Voicemeeter Out B1` |
| `SILENCE_THRESHOLD_MS` | VAD pause duration indicating utterance completion (ms) | All | `500` |
| `MAX_SILENCE_FLUSH_MS` | Failsafe silence threshold to force chunk transcription (ms) | All | `2000` |
| `MIN_LOCAL_CHUNK_S` | Minimum audio length before chunk finalization on natural pause (s) | Local / RunPod | `3.5` |
| `INTERIM_CHUNK_S` | Interval between interim transcription dispatches (seconds) | Local / RunPod | `5.0` |
| `MAX_LOCAL_CHUNK_S` | Maximum buffer duration cap before forced transcription (s) | Local / RunPod | `25.0` |
| `DEMO_FILE_PATH` | Path to sample WAV file for demo playback | All | `2025-12-14-1000.wav` |
| `VBAN_STREAM_NAME` | Target stream name for incoming VBAN UDP packets | All | `ringleiding` |

---

## 🧪 Running Tests

The project includes an automated test suite of **165 tests** covering audio framing, VAD chunking, anti-hallucination sliding-window algorithms, configuration loading, deployment contracts, and real-world streaming simulations.

Run the entire test suite via Python's standard `unittest` framework:

```bash
# Standard test discovery:
python3 -m unittest discover -s tests -p "test_*.py"

# Verbose output showing all 165 test cases:
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Test Suite Organization

- **`tests/test_deployment_contract.py`**:
  - Automated deployment, FUSE storage isolation, version manifest, and self-healing contracts.
  - Verifies FUSE object storage restrictions (safe byte-stream copying into local NVMe SSD).
  - Validates model caching and re-use checks to prevent redundant downloads across pod boots.
  - Bash syntax validity for `boot.sh` and `setup_once.sh`.
  - UI column header synchronization across `index.html` and `viewer.html`.

- **`tests/test_realworld_simulation.py`**:
  - Real-world stress testing with dirty log streams (500+ lines), error pattern escalation, step tracking [1/8]–[8/8], translation debouncing, and interim hallucination handling.
  - Rapid error pattern escalation into UI error banners without false positives on benign INFO logs.
  - Complete step tracking across phased boot stages `[1/8]` through `[8/8]`.
  - Interim translation debouncing and guaranteed sequential final completion.
  - Non-destructive interim hallucination handling preserving subsequent speech.

- **`tests/test_vad_audio.py`**:
  - WebRTC VAD 10ms/20ms/30ms framing, 48kHz/44.1kHz downmixing, 32-bit/8-bit to 16-bit PCM conversions, gain scaling, and RMS volume calculation.
  - WebRTC VAD strict frame sizing validation (480 samples / 960 bytes at 16kHz 16-bit mono).
  - Audio conversions: 48kHz and 44.1kHz stereo to 16kHz mono decimation.
  - Bit-depth conversions: 32-bit signed and 8-bit unsigned to 16-bit signed PCM.
  - Gain scaling with clipping bounds `[-32768, 32767]`, and RMS calculation.

- **`tests/test_anti_hallucination.py`**:
  - Sliding-window prefix overlap trimming, repetition loop suppression, and glossary/credit filtering.
  - Sliding-window prefix overlap trimming (`_transcribe_buffer` tail-head matching).
  - Suppression of 100% prefix leakage and identical repetition loops.
  - Filtering of isolated glossary phrases and low unique word ratio hallucinations.

- **`tests/test_config_modes.py`**:
  - PIPELINE_MODE resolution, language parsing, and environment configuration fallback logic.
  - PIPELINE_MODE resolution (`local`, `cloud`, `runpod`) and case-insensitivity.
  - Language code parsing and BCP-47 primary extraction.
  - Environment variable overrides and fallback defaults.

- **`tests/test_mock_pipeline.py`**:
  - Dynamic chunking state machine, pre-roll silence buffer capping, noise click rejection, and forced buffer flushing.
  - Pre-roll silence buffer capping (1.0s limit).
  - Force-flush on prolonged silence (`MAX_SILENCE_FLUSH_MS`).
  - Noise click rejection (< 90ms / 3 speech frames).
  - Buffer hard-cap transcription trigger (`MAX_LOCAL_CHUNK_S`).

- **`tests/test_core_package.py`**:
  - Modular architectural boundaries, engine isolation, RunPod REST/GraphQL API interactions, and telemetry condensation.
  - Headless audio ingestion and silence generator test fixtures.
  - Pod provisioning retry backoff, exponential delays, and quota policy error handling.

- **`tests/test_golden_contract.py`**:
  - Golden audio verification ensuring zero transcription degradation against baseline fixtures.
  - Accurate Dutch and multilingual transcription validation against known audio ground truths.

- **`tests/test_live_cloud_contract.py`**:
  - Contract verification for Google Cloud STT streaming and Translation client integrations.
  - Service account credential loading, client streaming mock handshakes, and stability gating.

- **`tests/test_live_cloud_e2e_audio_pipeline.py`**:
  - End-to-end cloud pipeline simulation over real loopback sockets verifying full-segment revision.
  - Multi-column translation broadcasts and cursive interim subtitle updates.

- **`tests/test_live_pipeline_integration.py`**:
  - Live audio capture integration tests verifying device routing and frame dispatch.
  - CoreAudio / WASAPI stream lifecycle and restart behavior upon stop/start cycles.

---

## 🌐 URLs & Access

When the application starts, it serves an administrative dashboard, a live subtitle viewer, and a telemetry health interface over HTTP and WebSockets:

| Interface | URL | Purpose |
|---|---|---|
| **Admin Panel** | `http://localhost/` or `http://localhost:8000/` | Full control: select audio input, switch languages, adjust gain, view volume VU meter, start/stop session. |
| **Live Viewer** | `http://localhost/viewer` or `http://ondertitels.local/viewer` | Clean, high-contrast, responsive subtitle view optimized for tablets, phones, or sanctuary projector screens. |
| **Telemetry Dashboard** | `http://localhost/telemetry` or `http://localhost:8000/telemetry` | Real-time cloud container telemetry, phased boot logs `[1/8]`–`[8/8]`, GPU hardware stats, STT/translation latencies, and WebSocket connection status. |

*Note: If port 80 is restricted or in use, the server automatically binds to port 8000.*

---

## 📦 Standalone Build (Windows EXE)

A standalone Windows executable can be generated using PyInstaller without requiring a local Python installation on the target production computer:

```bash
pip install pyinstaller
python build.py
```

The resulting executable will be placed in the `dist/` directory. Ensure your `.env` file and any glossary files (`SpGlos.txt`) reside alongside `app.exe`.

---

## 🔧 Troubleshooting

- **Client Docker Warning**:
  - Ondertitels does **not** need Docker on the client PC. If you see warnings or documentation mentioning Docker Desktop, note that the client runs 100% native Python.
- **Microphone Reads Complete Silence (RMS=0)**:
  - *macOS*: Ensure your terminal has microphone permissions under **System Settings → Privacy & Security → Microphone**.
  - *Windows*: Ensure VoiceMeeter is active and BUS B1 is receiving audio from your mixer or default capture device.
- **RunPod Connection Refused**:
  - Ensure outbound ports `443` and `30000-65535` are allowed in Windows Firewall.
  - Verify that `RUNPOD_API_KEY` is valid and that pods are running or auto-deploy is enabled.
  - Check the Telemetry dashboard at `http://localhost:8000/telemetry` to inspect live container logs and boot progress.
- **Google Cloud STT Credentials Error**:
  - Verify that `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON key file and that the Speech-to-Text and Cloud Translation APIs are enabled in your Google Cloud Console.

---

## 📄 License
Private and confidential. Developed for Live Subtitles & Translation ("Ondertitels").
