# Live Subtitles & Translation

Real-time speech-to-text subtitling with simultaneous translation into two target languages and two chained back-translations. This application supports a hybrid architecture: it can run **entirely locally** on Apple Silicon using MLX-Whisper, or on **Windows** leveraging **RunPod Cloud** for high-performance GPU acceleration.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/License-Private-red)

## 🌟 Features

- **Cross-Platform Support** — Optimized for both macOS (Local MLX) and Windows (Cloud RunPod).
- **5-Column Display** — Source audio, two direct translations, and two chained back-translations.
- **Intelligent Pod Management** — Fully automated, cost-saving lifecycle management for RunPod cloud servers.
- **Advanced Audio Processing** — WebRTC VAD with smart silence detection, force-flushing, and anti-hallucination filters.

---

## 🏗 Architecture

```text
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Source    │────▶│  AudioStream     │────▶│  STT Engine     │
│  (Mic / B1)     │     │  (PyAudio/ASIO)  │     │ (MLX or Faster) │
└─────────────────┘     └──────────────────┘     └────────┬────────┘
                                                          │
                        ┌──────────────────┐              │
                        │  WebSocket       │◀─────────────┘
                        │  (FastAPI)       │
                        └───────┬──────────┘
                                │
              ┌─────────────────┼─────────────────┐
              ▼                 ▼                 ▼
       ┌────────────┐   ┌────────────┐   ┌────────────┐
       │  Source     │   │  Ollama    │   │  Ollama    │
       │  Text (NL)  │   │  Translate │   │  Translate │
       │             │   │  (EN)      │   │  (RU)      │
       └────────────┘   └────────────┘   └────────────┘
```

---

## 🚀 Windows & RunPod Setup (Cloud STT)

This pipeline is optimized for Windows machines connected to professional mixers (like the Behringer X-32) using **VoiceMeeter**.

### 1. Prerequisites (Windows)
- **Python 3.12**
- **VoiceMeeter** (Standard, Banana, or Potato)
- **RunPod Account** with an active API Key

### 2. Windows Firewall Configuration
Run these commands in **PowerShell (as Administrator)** to ensure your local viewer and connections work flawlessly:

```powershell
# Allow Connection to RunPod (Outbound) - Note: WSS Proxy traverse firewalls automatically
New-NetFirewallRule -DisplayName "RunPod Outbound" -Direction Outbound -Action Allow -Protocol TCP -RemotePort 443,30000-65535

# Allow Viewer Access (Inbound) - So tablets/phones can access the web UI
New-NetFirewallRule -DisplayName "Ondertitels Viewer" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 80,8000
```

### 3. Audio Routing & Start
- In **VoiceMeeter**, route your mixer input to **BUS B1**.
- Start the server: `python app.py` (or use the standalone `app.exe`)
- The Python script will automatically detect and select `Voicemeeter Out B1` on startup.

---

## 🍎 macOS Setup (Local STT)

Run the entire pipeline locally without internet requirements, optimized for M1/M2/M3/M4 chips.

### 1. Prerequisites (Mac)
- **Python 3.10+**
- **Ollama** installed and running
- **PortAudio**: `brew install portaudio`

### 2. Installation
```bash
pip3 install -r requirements.txt
brew install ollama
ollama pull aya-expanse:8b
ollama serve
```

### 3. Start
```bash
python3 app.py
```

---

## 🤖 Intelligent Pod Lifecycle (RunPod)

To prevent runaway costs and keep your account clean, the backend uses a smart deployment sequence when `AUTO_DEPLOY_RUNPOD=true`:

1. **Auto-Resume**: The script first searches for any paused (`EXITED`) pods in your account. If found, it attempts to wake them up.
2. **Auto-Terminate**: If waking up fails (e.g., the RTX 4090 is out of stock in that specific datacenter), the script *terminates* the dead pod to keep your dashboard tidy.
3. **Auto-Deploy**: If no pods can be resumed, it automatically rents a fresh, brand new pod and waits for it to boot.
4. **Auto-Shutdown**: The RunPod server monitors its own uptime. After the configured `MAX_RUN_TIME_SEC` (default 90 mins), it executes a self-termination command to RunPod's GraphQL API, saving your wallet.

---

## 🐳 Docker & RunPod Deployment

To run the backend on RunPod, build and push the Docker image.

### 1. Build and Push the Image
```bash
docker build --platform linux/amd64 -t koenhu/runpod-vertaal:v9 .
docker push koenhu/runpod-vertaal:v9
```

### 2. RunPod Template Configuration
- **Container Image**: `koenhu/runpod-vertaal:v9`
- **Expose Port**: `8000` (HTTP/WS)
- **Environment Variables**:
  - `WHISPER_MODEL`: `large-v3` (CRITICAL for high quality)
  - `MAX_RUN_TIME_SEC`: `5400` (Auto-shutdown after 90 mins)

---

## 🛠 Configuration (.env)

Create a `.env` file in the project root:

| Variable | Description | Default |
|---|---|---|
| `PIPELINE_MODE` | `local` (Mac) or `runpod` (Windows) | `local` |
| `AUTO_DEPLOY_RUNPOD` | Enable intelligent pod resume/deploy | `true` |
| `RUNPOD_API_KEY` | Your RunPod API key | *(required for runpod mode)* |
| `DEFAULT_DEVICE_NAME` | Audio device to auto-select on startup | `Voicemeeter Out B1` |
| `MAX_RUN_TIME_SEC` | Auto-shutdown RunPod after X seconds | `5400` |
| `WHISPER_MODEL` | Local MLX model or RunPod model | `mlx-community/whisper-small-mlx` |
| `SOURCE_LANG` | Speech language | `nl-NL` |
| `TARGET_LANG_1` | First translation target | `en` |
| `TARGET_LANG_2` | Second translation target | `ru` |
| `ENABLE_TRANS_3` | Enable 3rd chained translation | `false` |
| `ENABLE_TRANS_4` | Enable 4th chained translation | `false` |

---

## 🌐 URLs & Access

Once the script is running, the interface is accessible via the local network. 
*Note: If the server cannot bind to port `80`, it will fall back to port `8000`.*

| Interface | URL | Purpose |
|---|---|---|
| **Admin Panel** | `http://localhost/` | Full control: Select audio source, change languages, pause/stop. |
| **Live Viewer** | `http://ondertitels.local/live` | Read-only, clean subtitles for tablets, smartphones, or projectors. |

---

## 🧠 Advanced Audio Processing

The app incorporates several intelligent safeguards to ensure stable subtitles and prevent Whisper AI from structurally omitting sentences or crashing into loops:

- **Silence Detection & Dynamic Chunking**:  
  Audio is continuously collected as long as the speaker talks. Under the hood, WebRTC Voice Activity Detection (VAD) monitors for micro-pauses. The engine aggressively capitalizes on any minor breath (≥ 400ms) to safely split and translate the audio without cutting words in half. 

- **Force-Flushing & Short Fragments**:  
  If a speaker provides an exceptionally short phrase (e.g. *"amen"*) and then remains perfectly quiet, a failsafe eventually forces a transcription. Extremely short microphone clicks (< 90ms) that accidentally trip the VAD are automatically dropped as noise, keeping the GPU buffer pure.

- **Anti-Hallucination & Prompt Sliding Trim**:  
  Because Whisper can suffer from "amnesia" between chunks, the backend employs **Prompt Context Sliding** — injecting the final 200 characters of the preceding audio chunk as a direct memory cue to the AI models. To combat the reverse side-effect (where Whisper sometimes hallucinate that memory cue out loud during absolute silence), a real-time sliding-window filter surgically detects repeated exact-text loops and strips hallucinated prefixes out of the final text before translation.

---

## 📦 Standalone Build (Windows EXE)

The standalone build system is fully operational. If you prefer to run the application as a standalone executable (`.exe`) without needing to invoke Python directly, you can compile it using the included build script. This bundles the entire Python environment into one file.

1. Install PyInstaller:
   ```bash
   pip install pyinstaller
   ```
2. Run the custom build script:
   ```bash
   python build.py
   ```
3. The newly compiled `app.exe` will appear in the `dist` folder.
   *Note: Always keep your `.env` file in the same directory as `app.exe` so it can load your API keys and preferred devices.*

---

## 🔧 Troubleshooting

### Microphone reads silence (RMS=0)
- **Mac**: Grant microphone permission to your terminal app in *System Settings → Privacy & Security → Microphone*.
- **Windows**: Check if VoiceMeeter is running and BUS B1 is actively receiving audio.

### Server won't stop
Use the **⏹ Stop** button in the web interface, or press `Ctrl+C` in the terminal.

### RunPod Connection Refused
Check your Windows Firewall rules. Ensure your network profile is set to **Private** and that Outbound port `443` and `30000-65535` are allowed.

---

## Credits
This software was developed with **AI Google Antigravity**, utilizing **Claude 3.5 Sonnet** and **Gemini 1.5 Pro** for cross-platform migration and low-latency audio engineering.
