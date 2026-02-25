# Live Subtitles & Translation

Real-time speech-to-text subtitling with simultaneous translation into two target languages. Runs **entirely locally** on Apple Silicon using MLX-Whisper for transcription and Ollama for translation — no cloud APIs, no internet required.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20Apple%20Silicon-lightgrey)
![License](https://img.shields.io/badge/License-Private-red)

## Features

- **Local Speech-to-Text** — MLX-Whisper optimized for Apple Silicon (M1/M2/M3/M4)
- **Local Translation** — Ollama LLM translation with configurable models (e.g., `aya-expanse:8b`)
- **Fully Offline** — No cloud APIs, no internet connection needed after initial model download
- **Voice Activity Detection** — WebRTC VAD with intelligent silence detection for natural sentence segmentation
- **Multiple Audio Sources** — Live microphone input or WAV file playback
- **Automatic Resampling** — WAV files in any format (stereo, 44.1kHz, 48kHz, etc.) are automatically converted to 16kHz mono for STT
- **Web Interface** — Three-column live display (source + 2 translations) with volume indicator
- **Device Switching** — Switch between microphones and audio files during a session
- **Session Limit** — 30-minute auto-pause safety timer

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Source    │────▶│  AudioStream     │────▶│  MLX-Whisper    │
│  (Mic / WAV)    │     │  (PyAudio + VAD) │     │  (Batch STT)    │
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

## Prerequisites

- Python 3.10+
- Apple Silicon Mac (M1/M2/M3/M4) — required for MLX-Whisper
- [Ollama](https://ollama.com/download) installed and running
- PortAudio (for PyAudio): `brew install portaudio`

## Installation

```bash
# Clone the repository
git clone https://github.com/koen5503/Vertaal.git
cd Vertaal

# Install Python dependencies
pip3 install -r requirements.txt

# Install and start Ollama
brew install ollama
ollama pull aya-expanse:8b
ollama serve   # keep running in a separate terminal

# Configure environment
cp .env.example .env
# Edit .env with your preferences
```

## Configuration

Create a `.env` file in the project root:

```env
WHISPER_MODEL=mlx-community/whisper-small-mlx
OLLAMA_MODEL=aya-expanse:8b
OLLAMA_URL=http://localhost:11434
SOURCE_LANG=nl-NL
TARGET_LANG_1=en
TARGET_LANG_2=ru
DEMO_FILE_PATH=BV.wav
TRANSLATE_INTERVAL_SEC=15
SILENCE_THRESHOLD_MS=300
```

| Variable | Description | Example |
|---|---|---|
| `WHISPER_MODEL` | MLX-Whisper model (HuggingFace repo) | `mlx-community/whisper-small-mlx`, `mlx-community/whisper-large-v3-turbo` |
| `OLLAMA_MODEL` | Ollama translation model | `aya-expanse:8b`, `qwen2.5:3b`, `llama3.1:8b` |
| `OLLAMA_URL` | Ollama server URL | `http://localhost:11434` |
| `SOURCE_LANG` | Source language (BCP-47 code) | `nl-NL`, `en-US` |
| `TARGET_LANG_1` | First translation target language | `en`, `fr`, `de` |
| `TARGET_LANG_2` | Second translation target language | `ru`, `uk`, `ar` |
| `DEMO_FILE_PATH` | Default WAV file for demo playback | `BV.wav` |
| `SILENCE_THRESHOLD_MS` | Silence duration to trigger transcription | `300`, `400` |

## Usage

### Starting the Server

```bash
python3 app.py
```

The server starts on `http://0.0.0.0:8000` (all network interfaces) and automatically opens your browser. The Whisper model is downloaded automatically on first run.

### Admin Interface (`/`)

Full control panel with audio source selection, language settings, and playback controls.

| Control | Function |
|---|---|
| **Source** | Select source speech language |
| **Left / Right** | Select target translation languages |
| **Input** | Switch between microphone, demo file, or WAV files |
| **🔄** | Rescan available audio input devices |
| **Pause / Resume** | Pause/resume transcription |
| **⏹ Stop** | Gracefully shut down the server |

### Read-Only Viewer (`/live`)

A stripped-down, read-only page for other devices on the local network. Shows the same 3-column subtitle display without any controls.

```
http://<your-mac-ip>:8000/live
```

Find your Mac's IP with: `ipconfig getifaddr en0`

## Supported Languages

### Source (Speech-to-Text)

- Dutch (nl-NL)
- English (en-US)

### Target (Translation)

- English, French, German, Russian, Ukrainian, Farsi, Arabic, Spanish, Portuguese, Italian, Chinese, Japanese, Korean, Turkish, Polish

## Technical Details

- **Audio Format**: 16kHz, 16-bit, mono PCM (LINEAR16)
- **Frame Duration**: 30ms (480 samples per chunk)
- **VAD**: WebRTC Voice Activity Detection (Mode 1)
- **STT**: Batch transcription triggered by silence threshold — audio accumulated in-memory as numpy array
- **Translation**: Async Ollama calls (`AsyncClient`) with `temperature=0` for deterministic output
- **Session Limit**: 30 minutes auto-pause
- **WebSocket**: Real-time bidirectional communication between server and browser
- **Resampling**: Automatic linear interpolation for non-16kHz WAV files

## Project Structure

```
Vertaal/
├── app.py                 # Main application (server, audio, STT, translation)
├── templates/
│   ├── index.html         # Admin interface (full controls)
│   └── viewer.html        # Read-only viewer for LAN clients (/live)
├── build.py               # PyInstaller build script
├── app.spec               # PyInstaller spec file
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (not in git)
├── .env.example           # Example configuration
└── *.wav                  # Demo audio files
```

## Troubleshooting

### Microphone reads silence (RMS=0)

- Grant microphone permission to your terminal app in *System Settings → Privacy & Security → Microphone*

### Ollama connection errors

- Ensure Ollama is running: `ollama serve`
- Ensure the model is pulled: `ollama pull aya-expanse:8b`
- Check the URL matches `OLLAMA_URL` in `.env`

### WAV file not found

WAV files placed in the same directory as the application are automatically found.

### Server won't stop

Use the **⏹ Stop** button in the web interface, or press `Ctrl+C` in the terminal.

---

## Credits

This software was entirely written with **AI Google Antigravity**, using the models **Claude Opus 4.6** and **Gemini 3.1 Pro**.
