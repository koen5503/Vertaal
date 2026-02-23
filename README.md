# Live Subtitles & Translation

Real-time speech-to-text subtitling with simultaneous translation into two target languages. Built for live events, meetings, and presentations where multilingual accessibility is needed.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Platform](https://img.shields.io/badge/Platform-macOS%20%7C%20Windows-lightgrey)
![License](https://img.shields.io/badge/License-Private-red)

## Features

- **Real-time Speech-to-Text** — Google Cloud Speech-to-Text with streaming recognition
- **Dual Translation** — Simultaneous translation into two configurable target languages (Google Cloud Translate)
- **Voice Activity Detection** — WebRTC VAD with intelligent silence detection for natural sentence segmentation
- **Multiple Audio Sources** — Live microphone input or WAV file playback
- **Automatic Resampling** — WAV files in any format (stereo, 44.1kHz, 48kHz, etc.) are automatically converted to 16kHz mono for STT
- **Web Interface** — Three-column live display (source + 2 translations) with volume indicator
- **Device Switching** — Switch between microphones and audio files during a session
- **Cross-Platform** — Runs on macOS and Windows, both as development server and standalone executable
- **Cost Control** — Automatic pause after 30 minutes to manage cloud API costs
- **Standalone Build** — PyInstaller packaging for single-file executable distribution

## Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Audio Source    │────▶│  AudioStream     │────▶│  Google STT     │
│  (Mic / WAV)    │     │  (PyAudio + VAD) │     │  (Streaming)    │
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
       │  Source     │   │  Google    │   │  Google    │
       │  Text (NL)  │   │  Translate │   │  Translate │
       │             │   │  (EN)      │   │  (RU)      │
       └────────────┘   └────────────┘   └────────────┘
```

## Prerequisites

- Python 3.10+
- Google Cloud project with enabled APIs:
  - Cloud Speech-to-Text
  - Cloud Translation
- Service account JSON key file
- PortAudio (for PyAudio):
  - macOS: `brew install portaudio`
  - Windows: included with PyAudio wheel

## Installation

```bash
# Clone the repository
git clone https://github.com/koen5503/Vertaal.git
cd Vertaal

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your Google Cloud credentials and preferences
```

## Configuration

Create a `.env` file in the project root:

```env
GOOGLE_APPLICATION_CREDENTIALS=your-service-account-key.json
GOOGLE_PROJECT_ID=your-project-id
SOURCE_LANG=nl-NL
TARGET_LANG_1=en
TARGET_LANG_2=ru
DEMO_FILE_PATH=BV.wav
```

| Variable | Description | Example |
|---|---|---|
| `GOOGLE_APPLICATION_CREDENTIALS` | Path to Google Cloud service account JSON | `my-project-key.json` |
| `GOOGLE_PROJECT_ID` | Google Cloud project ID | `my-project-123` |
| `SOURCE_LANG` | Source language (BCP-47 code) | `nl-NL`, `en-US` |
| `TARGET_LANG_1` | First translation target language | `en`, `fr`, `de` |
| `TARGET_LANG_2` | Second translation target language | `ru`, `uk`, `ar` |
| `DEMO_FILE_PATH` | Default WAV file for demo playback | `BV.wav` |

## Usage

### Development Server

```bash
python3 app.py
```

The application starts a local web server at `http://127.0.0.1:8000` and automatically opens your browser.

### Standalone Build

```bash
python build.py
```

Creates a single executable in `dist/app` (macOS) or `dist/app.exe` (Windows) with all resources bundled.

### Web Interface Controls

| Control | Function |
|---|---|
| **Source** | Select source speech language |
| **Left / Right** | Select target translation languages |
| **Input** | Switch between microphone, demo file, or WAV files |
| **🔄** | Rescan available audio input devices |
| **Pause / Resume** | Pause/resume transcription |
| **⏹ Stop** | Gracefully shut down the server |

## Supported Languages

### Source (Speech-to-Text)
- Dutch (nl-NL)
- English (en-US)

### Target (Translation)
- English, French, German, Russian, Ukrainian, Farsi, Arabic

## Technical Details

- **Audio Format**: 16kHz, 16-bit, mono PCM (LINEAR16)
- **Frame Duration**: 30ms (480 samples per chunk)
- **VAD**: WebRTC Voice Activity Detection (Mode 1)
- **Silence Threshold**: 400ms with stability > 0.8 triggers segment finalization
- **Session Limit**: 30 minutes auto-pause for cost control
- **WebSocket**: Real-time bidirectional communication between server and browser
- **Resampling**: Automatic linear interpolation for non-16kHz WAV files

## Project Structure

```
Vertaal/
├── app.py                 # Main application (server, audio, STT, translation)
├── templates/
│   └── index.html         # Web interface (3-column subtitle display)
├── build.py               # PyInstaller build script
├── app.spec               # PyInstaller spec file
├── requirements.txt       # Python dependencies
├── .env                   # Configuration (not in git)
├── .env.example           # Example configuration
└── *.wav                  # Demo audio files
```

## Troubleshooting

### Microphone reads silence (RMS=0)
- **macOS**: Grant microphone permission to your terminal app in *System Settings → Privacy & Security → Microphone*
- **Windows**: Check microphone access in *Settings → Privacy → Microphone*

### WAV file not found in standalone build
WAV files placed in the same directory as the executable are automatically found. Bundled files (via `app.spec`) are included in the executable.

### Server won't stop
Use the **⏹ Stop** button in the web interface, or press `Ctrl+C` in the terminal.

---

## Credits

This software was entirely written with **AI Google Antigravity**, using the models **Claude Opus 4.6** and **Gemini 3.1 Pro**.
