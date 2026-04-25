# RunPod Remote Execution Implementation Plan

This document outlines the phased development plan and exact prompts to provide to the development agent. The goal is to safely extend `app.py` to support a third `PIPELINE_MODE="runpod"`, retaining local audio capture while offloading STT and translation to a remote RunPod instance.

## User Review Required

Please review the proposed architecture and the series of prompts below. Once approved, you can feed these prompts one by one to the dedicated development agent.

## Proposed Changes / Architecture
When `PIPELINE_MODE=runpod`:
1. **Local Audio Capture**: `app.py` continues to handle PyAudio, VBAN, or WAV recordings exactly as it does now.
2. **Audio Streaming**: Instead of passing audio arrays to MLX-Whisper or Google Cloud, `app.py` establishes a WebSocket or HTTP streaming connection to the RunPod endpoint.
3. **RunPod Server**: A separate lightweight Python server (`runpod_server.py`) running in a RunPod Docker container receives the audio. Since RunPod uses NVIDIA GPUs, it will use `faster-whisper` (instead of Apple Silicon `mlx-whisper`) and Ollama for translation.
4. **Result Callback**: The RunPod server streams interim and final text back to the local `app.py`, which broadcasts it to the web UI exactly like the local mode.

---

## Phase 1: RunPod Server Component Development

**Prompt 1: RunPod Server Initialization**
> "Review `app.py` to understand the payload structures for transcription (`col1_text`) and translation (`update_translation`). Then, build a completely standalone FastAPI application named `runpod_server.py`. This server will run on a remote RunPod instance (Ubuntu/NVIDIA). It must expose a WebSocket endpoint `/stream` that accepts binary audio chunks (16kHz mono). It should buffer audio chunks and use `faster-whisper` (suitable for NVIDIA GPUs, not MLX) to transcribe, and `ollama` asyncio client to translate to `TARGET_LANG_1` and `TARGET_LANG_2`. It must yield JSON responses matching the exact structure expected by the viewer UI. Do not modify `app.py` yet."

**Prompt 2: Containerization & Deployment Setup**
> "Create a `Dockerfile` and `runpod_setup.md` specifically for the `runpod_server.py` application. The Dockerfile should install CUDA-compatible `faster-whisper`, `ollama`, and any required dependencies. The setup instructions should explain how to deploy this on RunPod, expose the Correct TCP port for WebSockets/HTTP, and retrieve the public RunPod URL. Ensure the environment is optimized for low-latency."

---

## Phase 2: Updating the Local Pipeline (`app.py`)

**Prompt 3: PIPELINE_MODE Extension**
> "Modify `app.py` and `.env.example`. Add support for `PIPELINE_MODE=runpod`. Add a new environment variable `RUNPOD_WSS_URL` to point to the remote RunPod WebSocket endpoint. Ensure that when 'runpod' is selected, MLX-Whisper and Google Cloud dependencies are elegantly bypassed so the app can still boot without them."

**Prompt 4: RunPod Streaming Logic**
> "Inside `TranscriptionEngine` in `app.py`, implement an `async def _run_runpod(self)` method. This method should behave identically to `_run_cloud()` but use the `websockets` library to establish a permanent connection to `RUNPOD_WSS_URL`. As the `audio_stream` reads chunks, pass them via the websocket. Listen for incoming JSON messages from the RunPod server and immediately pipe them to `self.broadcast()` so the frontend updates in real-time. Ensure robust connection handling (e.g., reconnects or graceful failures if RunPod is down)."

---

## Phase 3: Testing & Validation

**Prompt 5: Mock testing for app.py**
> "Create a mock websocket server script named `mock_runpod.py` that listens on `ws://localhost:8888`. It should receive binary audio and simulate transcription by occasionally sending back dummy JSON STT payloads and dummy translation payloads. Use this to verify that your changes to `app.py` correctly handle the `RunPod` mode and update the UI accordingly without needing to spool up a real GPU instance."

**Prompt 6: Regression & Integration Testing**
> "Review `app.py` after the RunPod integration. Verify that no changes broke the `PIPELINE_MODE=local` and `PIPELINE_MODE=cloud` functionality. Specifically, verify that VBAN support and device switching still function identically across all modes. Polish the codebase to remove testing artifacts."
