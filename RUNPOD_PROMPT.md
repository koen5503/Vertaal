# Task: Implement RunPod Execution Mode in OBS AI Subtitle Pipeline

## Context
The current repository contains an OBS AI Subtitle pipeline (`app.py`) with two operating modes defined by `PIPELINE_MODE`:
1. `local`: Uses `mlx_whisper` and a local `ollama` instance (works only on Apple Silicon Macs).
2. `cloud`: Uses Google Cloud STT and Translate.

## Objective
Your task is to add a third operating mode: **`runpod`**. This mode should retain the local audio capture (`pyaudio`) and VAD functionalities but offload the computationally heavy tasks (Speech-to-Text and LLM Translation) to a remote GPU server hosted on RunPod.

## Guidelines & Constraints
- **Do not break existing functionality**: Ensure `"local"` and `"cloud"` modes continue to work flawlessly.
- **Async Execution**: Follow the current async implementation pattern to keep the audio stream and UI responsive.
- **Robustness**: Implement proper error handling for network timeouts or RunPod cold starts.

---

## Step-by-Step Implementation Plan

### Phase 1: Configuration & Environment Setup
1. **Update Environment Files**: 
   - Modify `.env.example` to include new RunPod settings (e.g., `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_STT`, `RUNPOD_ENDPOINT_LLM`).
   - Add `runpod` as a valid option for `PIPELINE_MODE`.
2. **Update App Initialization**: 
   - Modify `TranscriptionEngine.__init__` in `app.py` to parse RunPod environment variables when `PIPELINE_MODE == "runpod"`.
   - Setup async HTTP clients (like `httpx.AsyncClient`) for communication with the RunPod endpoints.

### Phase 2: Remote Speech-to-Text (STT) Integration
1. **Implement `_transcribe_runpod`**:
   - Create a method that takes the local `audio_buffer` (collected via the existing VAD mechanism) and serializes it (e.g., in-memory WAV or raw PCM base64).
   - Send this audio data via HTTP POST to the RunPod STT endpoint (assume an OpenAI Whispers-compatible API or a custom RunPod serverless handler).
   - Parse the JSON response to extract the transcribed text.
2. **Apply Hallucination Filters**:
   - Ensure the existing anti-hallucination logic used in the `local` mode is applied to the text returned by RunPod.
3. **Execution Loop Integration**:
   - Implement `_run_runpod()` mirroring the logic of `_run_local()` but invoking `_transcribe_runpod` instead of `mlx_whisper`.

### Phase 3: Remote LLM Translation Integration
1. **Implement `_translate_runpod`**:
   - Create a method that mirrors `_translate_local` but targets the RunPod LLM endpoint.
   - If the RunPod instance is running Ollama or an OpenAI compatible inference server (e.g. vLLM), format the prompt to request direct translation without markdown or extra conversational text.
   - Send the STT text and return the translated string.
2. **Update Async Dispatcher**:
   - In `translate_text_async()`, add routing so `target_lang` queries use `_translate_runpod` when in RunPod mode.

### Phase 4: Server Setup Scripting
1. **Provide RunPod Server Instructions**:
   - Create a minimal `RUNPOD_SETUP.md` demonstrating how to run an OpenAI-compatible Whisper + LLM server on a RunPod instance (e.g., using `vllm` and `faster-whisper-server`). This serves as the remote counterpart for this integration.

### Phase 5: Thorough Testing Strategy
1. **Mock Testing**:
   - Create `test_runpod_mode.py`. Use mocking frameworks (e.g., `pytest-asyncio`, `respx`, or `unittest.mock`) to mock the RunPod HTTP endpoints. 
   - Write tests verifying that `_transcribe_runpod` correctly serializes the `bytearray` and that `_translate_runpod` processes network errors gracefully.
2. **Integration Verification Flow**:
   - Build a lightweight local dummy FastAPI server (`mock_runpod_server.py`) that implements the required RunPod endpoints and returns dummy transcriptions/translations.
   - Test the WebSocket broadcast mechanism by running `app.py` in `PIPELINE_MODE=runpod` against this dummy server.
   - Verify that the frontend UI effectively receives and renders the generated outputs.

Please follow this step-by-step plan exactly. Begin by analyzing `app.py` thoroughly, and then confirm when you are about to start Phase 1.
