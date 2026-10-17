# Live Subtitles & Translation ("Ondertitels")

Real-time speech-to-text subtitling with simultaneous multi-lingual translation and chained back-translations. The application is built for live events, church services, and presentations, featuring a unified FastAPI backend that flexibly supports three execution backends: **Local Apple Silicon (macOS MLX)**, **Google Cloud Speech-to-Text & Translation**, and **Remote NVIDIA GPU Acceleration (RunPod)**.

![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows%20%7C%20Linux-lightgrey)
![FastAPI](https://img.shields.io/badge/FastAPI-WebSocket-green)
![License](https://img.shields.io/badge/License-Private-red)

---

## 🌟 Key Features

- **3 Flexible Pipeline Modes** — Choose between 100% private on-device processing, ultra-low latency Google Cloud streaming, or high-throughput remote NVIDIA GPU acceleration on RunPod.
- **5-Column Subtitle Display** — Live source speech, two direct target translations, and two chained back-translations for real-time verification and multi-lingual reach.
- **Advanced Audio Processing & VAD** — WebRTC Voice Activity Detection (10ms, 20ms, or 30ms frames), gain normalization, automatic audio format resampling (48kHz/44.1kHz downmix to 16kHz mono), and RMS volume metering.
- **Anti-Hallucination Safeguards** — Dynamic sliding-window context memory cue with automated prefix-overlap trimming, repetition loops suppression, and glossary phrase leak filtering.
- **Multiple Audio Input Sources** — CoreAudio / ASIO / WASAPI microphones, VoiceMeeter virtual audio routing (BUS B1), network audio streaming via VBAN (UDP port 6980), or offline WAV file playback.
- **Automated RunPod Lifecycle Management** — Auto-discovery, auto-resume of paused pods, fallback GPU provisioning (RTX 3090, 4090, A4000, A5000), and auto-shutdown idle timer to minimize cloud compute expenses.

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
 │  48k/44.1k -> 16kHz Mono PCM │ 30ms Frames │ Gain Meter │
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

### The 3 Execution Backends

1. **Local Apple Silicon (`PIPELINE_MODE=local`)**
   - **STT**: `mlx-whisper` running natively on macOS Metal GPU/Neural Engine.
   - **Translation**: Local LLM served via `Ollama` (e.g., `aya-expanse:8b` or `qwen2.5:3b`).
   - **Characteristics**: 100% offline, zero cloud API costs, complete data privacy.

2. **Google Cloud (`PIPELINE_MODE=cloud`)**
   - **STT**: Google Cloud Speech-to-Text v1 streaming client with real-time stability tracking.
   - **Translation**: Google Cloud Translation API (v3/v2).
   - **Characteristics**: Lowest transcription latency for live spoken words; ideal for high-stakes broadcast environments with reliable internet.

3. **Remote NVIDIA GPU on RunPod (`PIPELINE_MODE=runpod`)**
   - **STT**: `faster-whisper` running on NVIDIA GPUs (RTX 3090, RTX 4090, A4000) deployed in the cloud.
   - **Communication**: Bidirectional streaming over WebSocket (`runpod_server.py`).
   - **Characteristics**: Top-tier accuracy with large models (`large-v3`) while allowing lightweight client hardware (e.g., Windows church PC without an internal GPU).

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

### ⚡ Backend 3: Windows & RunPod Cloud STT

Optimized for Windows PCs connected to professional mixers (such as Behringer X-32) using **VoiceMeeter**.

1. **Prerequisites**:
   - Python 3.10+ (or use compiled `app.exe`)
   - [VoiceMeeter](https://voicemeeter.com/) (Standard, Banana, or Potato)
   - [RunPod](https://www.runpod.io/) account with API key
2. **Windows Firewall Configuration (PowerShell as Administrator)**:
   ```powershell
   New-NetFirewallRule -DisplayName "RunPod Outbound" -Direction Outbound -Action Allow -Protocol TCP -RemotePort 443,30000-65535
   New-NetFirewallRule -DisplayName "Ondertitels Viewer" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 80,8000
   ```
3. **Audio Routing**:
   Route your mixer input in VoiceMeeter to **BUS B1**.
4. **Configure `.env`**:
   ```env
   PIPELINE_MODE=runpod
   RUNPOD_API_KEY=your_runpod_api_key
   RUNPOD_WHISPER_MODEL=large-v3
   RUNPOD_GPU_ID=NVIDIA GeForce RTX 3090
   AUTO_DEPLOY_RUNPOD=true
   MAX_RUN_TIME_SEC=5400
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
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud Service Account JSON file | Cloud | `""` |
| `GOOGLE_PROJECT_ID` | Google Cloud project identifier | Cloud | `""` |
| `STABILITY_THRESHOLD` | Minimum Google STT stability metric to trigger translation | Cloud | `0.8` |
| `MIN_WORDS_FOR_STABILITY` | Minimum words in transcript before interim translation | Cloud | `5` |
| `WHISPER_MODEL` | Hugging Face model identifier for MLX-Whisper | Local | `mlx-community/whisper-small-mlx` |
| `OLLAMA_MODEL` | Ollama model tag for LLM translation | Local / RunPod | `qwen2.5:3b` |
| `OLLAMA_URL` | Ollama service endpoint | Local / RunPod | `http://localhost:11434` |
| `RUNPOD_API_KEY` | RunPod GraphQL API key | RunPod | `""` |
| `RUNPOD_WSS_URL` | Manual WebSocket URL override (skips pod discovery) | RunPod | `""` |
| `RUNPOD_TEMPLATE_ID` | Template ID configured with runpod server image | RunPod | `""` |
| `RUNPOD_VOLUME_ID` | Network volume attached to pod | RunPod | `""` |
| `RUNPOD_GPU_ID` | Preferred GPU type for pod rental | RunPod | `NVIDIA GeForce RTX 3090` |
| `RUNPOD_DATA_CENTER_ID` | Optional datacenter code (e.g. `EU-RO-1`) | RunPod | `""` |
| `RUNPOD_WHISPER_MODEL` | Whisper model loaded on remote pod (`small`, `large-v3`) | RunPod | `large-v3` |
| `AUTO_DEPLOY_RUNPOD` | Automatically resume or spin up pod on startup | RunPod | `false` |
| `MAX_RUN_TIME_SEC` | Idle timeout for RunPod server self-termination (seconds) | RunPod | `5400` |
| `DEFAULT_DEVICE_NAME` | Audio input device auto-selected on boot | All | `Voicemeeter Out B1` |
| `SILENCE_THRESHOLD_MS` | VAD pause duration indicating utterance completion (ms) | All | `400` |
| `MAX_SILENCE_FLUSH_MS` | Failsafe silence threshold to force chunk transcription (ms) | All | `2000` |
| `MIN_LOCAL_CHUNK_S` | Minimum audio length before transcribing (seconds) | Local / RunPod | `10.0` |
| `MAX_LOCAL_CHUNK_S` | Maximum buffer duration cap before forced transcription (s) | Local / RunPod | `25.0` |
| `MAX_SESSION_MINUTES` | Safety limit before auto-pausing pipeline (minutes) | All | `30` |
| `DEMO_FILE_PATH` | Path to sample WAV file for demo playback | All | `2025-12-14-1000.wav` |
| `VBAN_STREAM_NAME` | Target stream name for incoming VBAN UDP packets | All | `ringleiding` |

---

## 🧪 Running Tests

The project includes an automated regression test suite covering audio framing, VAD chunking, anti-hallucination sliding-window algorithms, configuration loading, and pipeline state machines.

Run the entire test suite via Python's standard `unittest` framework:

```bash
# Standard test discovery:
python3 -m unittest discover -s tests -p "test_*.py"

# Verbose output showing all test cases:
python3 -m unittest discover -s tests -p "test_*.py" -v
```

### Test Suite Organization

- **`tests/test_vad_audio.py`**:
  - WebRTC VAD 10ms, 20ms, and 30ms strict frame sizing validation (480 samples / 960 bytes at 16kHz 16-bit mono).
  - Audio conversions: 48kHz and 44.1kHz stereo to 16kHz mono decimation.
  - Bit-depth conversions: 32-bit signed and 8-bit unsigned to 16-bit signed PCM.
  - Gain scaling with clipping bounds `[-32768, 32767]`, and RMS calculation.

- **`tests/test_anti_hallucination.py`**:
  - Sliding-window prefix overlap trimming (`_transcribe_buffer` tail-head matching).
  - Suppression of 100% prefix leakage and identical repetition loops.
  - Filtering of isolated glossary phrases and low unique word ratio hallucinations.

- **`tests/test_config_modes.py`**:
  - PIPELINE_MODE resolution (`local`, `cloud`, `runpod`) and case-insensitivity.
  - Language code parsing and BCP-47 primary extraction.
  - Environment variable overrides and fallback defaults.

- **`tests/test_mock_pipeline.py`**:
  - Dynamic chunking state machine.
  - Pre-roll silence buffer capping (1.0s limit).
  - Force-flush on prolonged silence (`MAX_SILENCE_FLUSH_MS`).
  - Noise click rejection (< 90ms / 3 speech frames).
  - Buffer hard-cap transcription trigger (`MAX_LOCAL_CHUNK_S`).

---

## 🌐 URLs & Access

When the application starts, it serves both an administrative dashboard and a live viewer over HTTP and WebSockets:

| Interface | URL | Purpose |
|---|---|---|
| **Admin Panel** | `http://localhost/` or `http://localhost:8000/` | Full control: select audio input, switch languages, adjust gain, view volume VU meter, start/stop session. |
| **Live Viewer** | `http://localhost/viewer` or `http://ondertitels.local/viewer` | Clean, high-contrast, responsive subtitle view optimized for tablets, phones, or sanctuary projector screens. |

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

- **Microphone Reads Complete Silence (RMS=0)**:
  - *macOS*: Ensure your terminal has microphone permissions under **System Settings → Privacy & Security → Microphone**.
  - *Windows*: Ensure VoiceMeeter is active and BUS B1 is receiving audio from your mixer or default capture device.
- **RunPod Connection Refused**:
  - Ensure outbound ports `443` and `30000-65535` are allowed in Windows Firewall.
  - Verify that `RUNPOD_API_KEY` is valid and that pods are running or auto-deploy is enabled.
- **Google Cloud STT Credentials Error**:
  - Verify that `GOOGLE_APPLICATION_CREDENTIALS` points to a valid JSON key file and that the Speech-to-Text and Cloud Translation APIs are enabled in your Google Cloud Console.

---

## 📄 License
Private and confidential. Developed for Live Subtitles & Translation ("Ondertitels").
