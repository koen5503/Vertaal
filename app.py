import asyncio
import json
import logging
import os
import struct
import math
import time
import threading
import queue
import sys
from types import ModuleType
from dotenv import load_dotenv

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")
    return os.path.join(base_path, relative_path)

# Pass the resolved path to the .env file
env_path = resource_path(".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

# Mock pkg_resources for webrtcvad (if needed repeatedly)
if 'pkg_resources' not in sys.modules:
    class MockDistribution: version = '2.0.10'
    class MockPkgResources:
        @staticmethod
        def get_distribution(name): return MockDistribution()
    sys.modules['pkg_resources'] = MockPkgResources()

import webrtcvad
import pyaudio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

# --- Pipeline Mode ---
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "local").lower()  # "local" or "cloud"

if PIPELINE_MODE == "cloud":
    from google.cloud import speech
    from google.cloud import translate
    print("Pipeline mode: CLOUD (Google Cloud STT + Translate)")
else:
    import mlx_whisper
    import numpy as np
    from ollama import AsyncClient
    print("Pipeline mode: LOCAL (MLX-Whisper + Ollama)")

app = FastAPI()

# --- Audio Constants ---
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) # 480 samples


import wave

class AudioStream:
    """Captures audio from microphone or file in a background thread."""
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.vad = webrtcvad.Vad(1) # Mode 1: Standard
        self.queue = queue.Queue()
        self.running = False
        self.playback_paused = False
        self.thread = None
        
        # Default to demo file playback on startup
        self.current_device_index = "demo_file"
        self.stream = None
        print(f"Startup mode: Demo File ({os.getenv('DEMO_FILE_PATH', 'BV.wav')})")

    def _open_stream(self):
        # Close existing if open
        if hasattr(self, 'stream') and self.stream:
            self.stream.close()
            self.stream = None
            
        if self.current_device_index == "demo_file":
            print("Preparing to stream from Demo File")
            self.stream = None
            return
            
        print(f"Opening Stream on Device Index: {self.current_device_index}")
        try:
            self.stream = self.p.open(
                format=pyaudio.paInt16,
                channels=CHANNELS,
                rate=SAMPLE_RATE,
                input=True,
                input_device_index=self.current_device_index,
                frames_per_buffer=CHUNK_SIZE,
            )
        except Exception as e:
            print(f"CRITICAL: Failed to open PyAudio stream on device {self.current_device_index}: {e}")
            self.stream = None

    def list_devices(self):
        """Uses existing PyAudio instance to scan for new devices and returns list."""
        devices = []
        try:
            count = self.p.get_device_count()
            for i in range(count):
                try:
                    info = self.p.get_device_info_by_index(i)
                    if info.get('maxInputChannels') > 0:
                        devices.append({"index": i, "name": info.get('name')})
                except Exception as e:
                    pass
        except Exception as e:
            print(f"Error getting device info: {e}")
                
        # Dynamically add all .wav files from CWD and bundle directory
        import glob
        wav_files = set(os.path.basename(f) for f in glob.glob("*.wav"))
        bundle_dir = resource_path(".")
        if bundle_dir != os.path.abspath("."):
            wav_files.update(os.path.basename(f) for f in glob.glob(os.path.join(bundle_dir, "*.wav")))
        for w in sorted(wav_files):
            devices.append({"index": f"file_{w}", "name": f"File: {w}"})
            
        return devices

    def change_device(self, index):
        """Switches input device by thoroughly resetting the stream."""
        print(f"Switching to Audio Device Index: {index}")
        normalized_index = index if index is not None else None
        
        # Allow re-selection of demo_file (actual file may have changed)
        if self.current_device_index == normalized_index and normalized_index != "demo_file":
            return
            
        self.current_device_index = normalized_index
        
        # Force running to False in case previous file loop already ended
        self.running = False
        
        # Stop reading and close streams (but don't terminate PyAudio yet)
        self.stop(terminate_pyaudio=False)
        
        # Flush the old audio queue
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
                
        # Recreate PyAudio to clear macOS CoreAudio sample rate lock
        try:
            self.p.terminate()
        except:
            pass
        self.p = pyaudio.PyAudio()
        
        # Reopen input stream on fresh instance
        self._open_stream()

    def start(self):
        if self.running: return
        self.running = True
        
        # Unconditionally call start_stream to wake up frozen Mac CoreAudio units
        if self.stream:
            try:
                self.stream.start_stream()
            except Exception as e:
                print(f"Failed to start stream: {e}")
                
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        """Background thread to continuously read audio."""
        if self.current_device_index == "demo_file":
            self._read_file_loop()
            return
            
        while self.running:
            try:
                if self.stream is None:
                    time.sleep(0.5)
                    continue
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.queue.put(data)
            except Exception as e:
                print(f"Audio Read Error: {e}")
                break

    def _resample_for_stt(self, data, src_rate, src_channels, src_width):
        """Convert audio data to 16kHz mono 16-bit LINEAR16 for STT."""
        # Unpack samples to 16-bit signed integers
        if src_width == 2:
            n = len(data) // 2
            samples = list(struct.unpack(f"<{n}h", data))
        elif src_width == 1:
            samples = [((b - 128) << 8) for b in data]
        elif src_width == 4:
            n = len(data) // 4
            raw = struct.unpack(f"<{n}i", data)
            samples = [s >> 16 for s in raw]
        else:
            return data
        
        # Stereo/multi-channel to mono
        if src_channels >= 2:
            mono = []
            for i in range(0, len(samples), src_channels):
                mono.append(sum(samples[i:i+src_channels]) // src_channels)
            samples = mono
        
        # Resample to 16kHz using linear interpolation
        if src_rate != SAMPLE_RATE:
            ratio = src_rate / SAMPLE_RATE
            new_len = int(len(samples) / ratio)
            resampled = []
            for i in range(new_len):
                src_pos = i * ratio
                idx = int(src_pos)
                frac = src_pos - idx
                if idx + 1 < len(samples):
                    val = int(samples[idx] * (1 - frac) + samples[idx + 1] * frac)
                else:
                    val = samples[idx] if idx < len(samples) else 0
                resampled.append(max(-32768, min(32767, val)))
            samples = resampled
        
        return struct.pack(f"<{len(samples)}h", *samples)

    def _read_file_loop(self):
        demo_name = os.getenv("DEMO_FILE_PATH", "2025-12-14-1000.wav")
        # Try bundled path first, then CWD for dynamically selected files
        file_path = resource_path(demo_name)
        if not os.path.exists(file_path):
            file_path = os.path.abspath(demo_name)
        if not os.path.exists(file_path):
            print(f"Error: Demo file not found at {file_path}")
            self.running = False
            return
            
        try:
            wf = wave.open(file_path, 'rb')
            src_rate = wf.getframerate()
            src_channels = wf.getnchannels()
            src_width = wf.getsampwidth()
            needs_resample = (src_rate != SAMPLE_RATE or src_channels != CHANNELS)
            print(f"Playing: {demo_name} (rate={src_rate}, channels={src_channels}, width={src_width}, resample={needs_resample})")
            
            out_stream = self.p.open(
                format=self.p.get_format_from_width(src_width),
                channels=src_channels,
                rate=src_rate,
                output=True
            )
            
            sleep_time = CHUNK_SIZE / src_rate
            data = wf.readframes(CHUNK_SIZE)
            
            while self.running and len(data) > 0:
                if self.playback_paused:
                    time.sleep(0.1)
                    continue
                    
                start_t = time.time()
                try:
                    out_stream.write(data)
                except Exception as e:
                    print(f"Write to PyAudio out failed: {e}")
                    break
                
                if len(data) < CHUNK_SIZE * src_width * src_channels:
                    data += b'\x00' * (CHUNK_SIZE * src_width * src_channels - len(data))

                # Resample for STT if needed, then queue
                if needs_resample:
                    stt_data = self._resample_for_stt(data, src_rate, src_channels, src_width)
                else:
                    stt_data = data
                self.queue.put(stt_data)
                data = wf.readframes(CHUNK_SIZE)
                
                elapsed = time.time() - start_t
                if elapsed < sleep_time * 0.9: 
                    time.sleep((sleep_time * 0.9) - elapsed)
                    
            try:
                out_stream.stop_stream()
                out_stream.close()
            except:
                pass
            wf.close()
        except Exception as e:
            print(f"File Read Error: {e}")
            
        print("File Demo Finished or Stopped")
        self.running = False

    def stop(self, terminate_pyaudio=False):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
            
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
                
        if terminate_pyaudio and hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except:
                pass

    def read_chunk(self):
        """Blocking get from queue."""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            print("WARNING: Audio Queue Empty (generating silence)")
            return b'\x00' * (CHUNK_SIZE * 2)

    def calculate_rms(self, data):
        count = len(data) // 2
        format = "<%dh" % count
        shorts = struct.unpack(format, data)
        sum_squares = sum(s * s for s in shorts)
        return int(math.sqrt(sum_squares / count))

    def apply_gain(self, data, gain):
        count = len(data) // 2
        shorts = struct.unpack(f"<{count}h", data)
        scaled = []
        for s in shorts:
            val = int(s * gain)
            if val > 32767: val = 32767
            if val < -32768: val = -32768
            scaled.append(val)
        return struct.pack(f"<{count}h", *scaled)


# --- Language code to full name mapping for Ollama translation prompt ---
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "ru": "Russian",
    "uk": "Ukrainian", "fa": "Farsi", "ar": "Arabic", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "pl": "Polish",
}


class TranscriptionEngine:
    """Manages STT and translation — supports both cloud (Google) and local (MLX-Whisper + Ollama) modes."""
    def __init__(self, broadcast_callback):
        self.pipeline_mode = PIPELINE_MODE

        # State (shared)
        self.source_lang = os.getenv("SOURCE_LANG", "nl-NL")
        self.target_lang_1 = os.getenv("TARGET_LANG_1", "en")
        self.target_lang_2 = os.getenv("TARGET_LANG_2", "fr")
        self.is_paused = False
        self.restart_required = False

        self.broadcast = broadcast_callback
        self.audio_stream = AudioStream()

        # Timing / VAD state
        self.silence_frames = 0
        self.speech_frames = 0
        self.current_seg_id = f"seg_{int(time.time()*1000)}"
        self.last_translate_time = 0
        self.translate_interval = int(os.getenv("TRANSLATE_INTERVAL_SEC", "10"))
        self.silence_threshold_ms = int(os.getenv("SILENCE_THRESHOLD_MS", "400"))
        self.translate_count = 0
        self.session_start_time = time.time()
        self.MAX_DURATION_SECONDS = 30 * 60  # 30 minutes

        # Load glossary if exists
        self.glossary_text = ""
        self.glossary_phrases = []
        glos_path = resource_path("SpGlos.txt")
        if os.path.exists(glos_path):
            with open(glos_path, "r", encoding="utf-8") as f:
                self.glossary_text = f.read().strip()
                # Split by comma for Google Cloud phrases
                self.glossary_phrases = [p.strip() for p in self.glossary_text.split(",") if p.strip()]
                print(f"Loaded glossary from {glos_path}: {len(self.glossary_phrases)} phrases")

        if self.pipeline_mode == "cloud":
            # --- Cloud mode setup ---
            self.credentials_path = resource_path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
            self.project_id = os.getenv("GOOGLE_PROJECT_ID")

            if not self.credentials_path or not os.path.exists(self.credentials_path):
                raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set or file does not exist. Check your .env file.")
            if not self.project_id:
                raise ValueError("GOOGLE_PROJECT_ID is not set. Check your .env file.")

            self.speech_client = speech.SpeechClient.from_service_account_json(self.credentials_path)
            self.translate_client = translate.TranslationServiceClient.from_service_account_json(self.credentials_path)
            self.parent = f"projects/{self.project_id}/locations/global"
            self.stability_threshold = float(os.getenv("STABILITY_THRESHOLD", "0.8"))
            self.min_words = int(os.getenv("MIN_WORDS_FOR_STABILITY", "5"))
            self.last_translated_text = ""
            self._setup_recognition_config()
            print(f"Cloud STT: Google Speech-to-Text")
            print(f"Cloud Translation: Google Cloud Translate")
        else:
            # --- Local mode setup ---
            self.whisper_model = os.getenv("WHISPER_MODEL", "mlx-community/whisper-small-mlx")
            self.ollama_model = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
            self.ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
            self.min_speech_frames = 10  # ~300ms minimum speech before transcribing
            self.audio_buffer = bytearray()
            print(f"Local STT model : {self.whisper_model}")
            print(f"Ollama model    : {self.ollama_model} @ {self.ollama_url}")

    # --- Cloud-only helpers ---

    def _setup_recognition_config(self):
        speech_contexts = []
        if self.glossary_phrases:
            speech_contexts.append(speech.SpeechContext(phrases=self.glossary_phrases))

        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.source_lang,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
            speech_contexts=speech_contexts,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
        )

    # --- Shared methods ---

    def get_input_devices(self):
        return self.audio_stream.list_devices()

    def update_config(self, config):
        print(f"Updating Config: {config}")

        if "paused" in config:
            was_paused = self.is_paused
            self.is_paused = config["paused"]
            self.audio_stream.playback_paused = self.is_paused
            if was_paused and not self.is_paused:
                self.session_start_time = time.time()

        if "source_lang" in config and config["source_lang"] != self.source_lang:
            self.source_lang = config["source_lang"]
            if self.pipeline_mode == "cloud":
                self._setup_recognition_config()
            self.restart_required = True

        if "target_lang_1" in config:
            self.target_lang_1 = config["target_lang_1"]

        if "target_lang_2" in config:
            self.target_lang_2 = config["target_lang_2"]

        if "device_index" in config:
            idx = config["device_index"]

            if isinstance(idx, str) and idx.startswith("file_"):
                file_name = idx.replace("file_", "", 1)
                os.environ["DEMO_FILE_PATH"] = file_name
                self.audio_stream.change_device("demo_file")
                self.restart_required = True
            elif idx == "demo_file":
                if self.audio_stream.current_device_index != "demo_file":
                    self.audio_stream.change_device("demo_file")
                    self.restart_required = True
            elif idx == "default" or idx is None:
                try:
                    default = self.audio_stream.p.get_default_input_device_info()
                    self.audio_stream.change_device(int(default['index']))
                except:
                    self.audio_stream.change_device(None)
                self.restart_required = True
            else:
                self.audio_stream.change_device(int(idx))
                self.restart_required = True

    # --- Translation ---

    async def translate_text_async(self, text, target_lang):
        """Translate text using cloud or local backend."""
        if not text or not target_lang:
            return ""
        if target_lang == self.source_lang.split("-")[0]:
            return text

        if self.pipeline_mode == "cloud":
            return await self._translate_cloud(text, target_lang)
        else:
            return await self._translate_local(text, target_lang)

    async def _translate_cloud(self, text, target_lang):
        """Google Cloud Translate."""
        loop = asyncio.get_running_loop()

        def _call():
            try:
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "contents": [text],
                        "mime_type": "text/plain",
                        "source_language_code": self.source_lang.split("-")[0],
                        "target_language_code": target_lang,
                    }
                )
                return response.translations[0].translated_text
            except Exception as e:
                print(f"Translation Error ({target_lang}): {e}")
                return text

        return await loop.run_in_executor(None, _call)

    async def _translate_local(self, text, target_lang):
        """Ollama AsyncClient translation."""
        lang_name = LANG_NAMES.get(target_lang, target_lang)
        source_lang_code = self.source_lang.split("-")[0]
        source_lang_name = LANG_NAMES.get(source_lang_code, source_lang_code)
        system_prompt = (
            f"Je bent een professionele, native vertaler. "
            f"Vertaal de volgende {source_lang_name}e tekst naar het {lang_name}. "
            f"Behoud de originele toon en betekenis perfect. "
            f"Geef UITSLUITEND de directe vertaling terug. "
            f"Geef geen uitleg, geen introductie, geen aanhalingstekens en geen markdown."
        )

        try:
            client = AsyncClient(host=self.ollama_url)
            response = await client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                options={"temperature": 0},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Translation Error ({target_lang}): {e}")
            return text

    async def handle_translation(self, seg_id, text, target_lang, col_key):
        """Background task to translate and update UI."""
        if not text:
            return
        translated = await self.translate_text_async(text, target_lang)
        msg = {
            "action": "update_translation",
            "id": seg_id,
            "col_key": col_key,
            "text": translated,
        }
        await self.broadcast(msg)

    # --- Main run loop (dispatches to cloud or local) ---

    async def run(self):
        """Main loop — dispatches to cloud or local STT engine."""
        print("Starting Audio Stream...")
        self.audio_stream.start()

        if self.pipeline_mode == "cloud":
            await self._run_cloud()
        else:
            await self._run_local()

    # --- Cloud STT (Google Streaming) ---

    async def _run_cloud(self):
        """Cloud mode: stream audio to Google STT."""
        loop = asyncio.get_event_loop()

        while True:
            if not self.audio_stream.running:
                await asyncio.sleep(0.5)
                continue

            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            self.restart_required = False
            stop_event = asyncio.Event()

            await loop.run_in_executor(None, self._run_google_stream_sync, stop_event, loop)

            if self.restart_required:
                print("Restarting stream due to config change...")
                if not self.audio_stream.running:
                    self.audio_stream.start()

            self.current_seg_id = f"seg_{int(time.time()*1000)}"

    def _run_google_stream_sync(self, stop_event, loop):
        """Blocking function to handle one Google Stream session."""
        generator_stop = False

        def generator():
            nonlocal generator_stop
            chunk_count = 0

            if not self.is_paused and chunk_count == 0:
                self.session_start_time = time.time()

            while not generator_stop:
                if self.restart_required or self.is_paused:
                    generator_stop = True
                    return

                if time.time() - self.session_start_time > self.MAX_DURATION_SECONDS:
                    print(f"TIMEOUT REACHED ({self.MAX_DURATION_SECONDS}s). Auto-pausing.")
                    self.is_paused = True
                    msg = {"action": "auto_paused", "reason": "30_min_limit"}
                    asyncio.run_coroutine_threadsafe(self.broadcast(msg), loop=loop)
                    stop_event.set()
                    return

                chunk_count += 1
                chunk = self.audio_stream.read_chunk()
                if not chunk:
                    generator_stop = True
                    break

                if chunk_count % 10 == 0:
                    rms = self.audio_stream.calculate_rms(chunk)
                    vol_msg = {"type": "volume", "rms": rms}
                    asyncio.run_coroutine_threadsafe(self.broadcast(vol_msg), loop=loop)

                is_speech = self.audio_stream.vad.is_speech(chunk, SAMPLE_RATE)
                if is_speech:
                    self.silence_frames = 0
                else:
                    self.silence_frames += 1

                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        try:
            responses = self.speech_client.streaming_recognize(
                config=self.streaming_config,
                requests=generator()
            )

            for response in responses:
                if not response.results: continue
                result = response.results[0]
                if not result.alternatives: continue

                transcript = result.alternatives[0].transcript
                stability = result.stability
                is_final = result.is_final

                print(f"STT: '{transcript}' | Words: {len(transcript.split())} | Stability: {stability:.2f} | Final: {is_final}")

                word_count = len(transcript.split())
                silence_ms = self.silence_frames * FRAME_DURATION_MS

                eff_stability = 1.0 if word_count < self.min_words else self.stability_threshold

                sentence_end = transcript.rstrip().endswith(('.', '!', '?'))

                status = "final" if is_final else "interim"

                if (silence_ms > self.silence_threshold_ms or sentence_end) and stability > eff_stability:
                    if not is_final:
                        reason = f"Punctuation '{transcript[-1]}'" if sentence_end else f"Silence {silence_ms}ms"
                        print(f"FORCE FINAL: '{transcript}' ({reason}) — waiting for Google final...")
                        generator_stop = True

                if is_final:
                    stop_event.set()

                msg = {
                    "id": self.current_seg_id,
                    "status": status,
                    "col1_text": transcript,
                }
                asyncio.run_coroutine_threadsafe(self.broadcast(msg), loop=loop)

                now = time.time()
                if is_final or (stability > eff_stability and now - self.last_translate_time >= self.translate_interval):
                    self.last_translate_time = now
                    self.translate_count += len(transcript) * 2
                    print(f"Translate ({self.translate_count} chars total): '{transcript[:50]}'..." if len(transcript) > 50 else f"Translate ({self.translate_count} chars total): '{transcript}'")
                    asyncio.run_coroutine_threadsafe(
                        self.handle_translation(self.current_seg_id, transcript, self.target_lang_1, "trans1"),
                        loop=loop
                    )
                    asyncio.run_coroutine_threadsafe(
                        self.handle_translation(self.current_seg_id, transcript, self.target_lang_2, "trans2"),
                        loop=loop
                    )

                if stop_event.is_set():
                    break

        except Exception as e:
            print(f"Stream Error: {e}")
            stop_event.set()

    # --- Local STT (MLX-Whisper batch) ---

    def _transcribe_buffer(self):
        """Convert accumulated audio buffer to numpy and run mlx-whisper."""
        if len(self.audio_buffer) < CHUNK_SIZE * 2:
            return None

        audio_np = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        whisper_lang = self.source_lang.split("-")[0]

        transcribe_kwargs = {
            "path_or_hf_repo": self.whisper_model,
            "language": whisper_lang,
        }
        if self.glossary_text:
            transcribe_kwargs["initial_prompt"] = self.glossary_text

        try:
            result = mlx_whisper.transcribe(
                audio_np,
                **transcribe_kwargs
            )
            text = result.get("text", "").strip()

            # --- Anti-Hallucination Filter ---
            if text:
                import string
                clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
                words = clean_text.split()

                if not words:
                    return None

                # 1. Repetition filter (e.g., "kerk kerk kerk" or "kerk kerk")
                if len(words) >= 2 and len(set(words)) == 1:
                    print(f"Filtered hallucination (repetitive): '{text}'")
                    return None
                if len(words) >= 4 and len(set(words)) <= 2:
                    print(f"Filtered hallucination (repetitive loop): '{text}'")
                    return None

                # 2. Isolated glossary word filter
                # When VAD catches a breath/noise, Whisper defaults to outputting 1-2 words from initial_prompt
                if self.glossary_phrases and len(words) <= 2:
                    glos_lower = [p.lower() for p in self.glossary_phrases]
                    # Check if any word exactly matches a glossary term
                    if any(w in glos_lower for w in words) or clean_text in glos_lower:
                        print(f"Filtered hallucination (isolated glossary term): '{text}'")
                        return None

            return text if text else None
        except Exception as e:
            print(f"Whisper Transcription Error: {e}")
            return None

    async def _run_local(self):
        """Local mode: accumulate audio, transcribe on silence via MLX-Whisper."""
        loop = asyncio.get_event_loop()
        chunk_count = 0

        while True:
            # Handle restart FIRST (device/file switch sets running=False)
            if self.restart_required:
                print("Restarting stream due to config change...")
                self.restart_required = False
                self.audio_buffer = bytearray()
                self.silence_frames = 0
                self.speech_frames = 0
                self.current_seg_id = f"seg_{int(time.time()*1000)}"
                if not self.audio_stream.running:
                    self.audio_stream.start()
                continue

            if not self.audio_stream.running:
                await asyncio.sleep(0.5)
                continue

            if self.is_paused:
                await asyncio.sleep(0.1)
                continue

            if time.time() - self.session_start_time > self.MAX_DURATION_SECONDS:
                print(f"TIMEOUT REACHED ({self.MAX_DURATION_SECONDS}s). Auto-pausing.")
                self.is_paused = True
                await self.broadcast({"action": "auto_paused", "reason": "30_min_limit"})
                continue

            chunk = await loop.run_in_executor(None, self.audio_stream.read_chunk)
            if not chunk:
                continue

            chunk_count += 1

            if chunk_count % 10 == 0:
                rms = self.audio_stream.calculate_rms(chunk)
                await self.broadcast({"type": "volume", "rms": rms})

            is_speech = self.audio_stream.vad.is_speech(chunk, SAMPLE_RATE)
            if is_speech:
                self.silence_frames = 0
                self.speech_frames += 1
            else:
                self.silence_frames += 1

            self.audio_buffer.extend(chunk)

            silence_ms = self.silence_frames * FRAME_DURATION_MS

            if silence_ms >= self.silence_threshold_ms and self.speech_frames >= self.min_speech_frames:
                buffer_duration_s = len(self.audio_buffer) / (SAMPLE_RATE * 2)
                print(f"Silence detected ({silence_ms}ms) — transcribing {buffer_duration_s:.1f}s of audio...")

                transcript = await loop.run_in_executor(None, self._transcribe_buffer)

                if transcript:
                    print(f"STT: '{transcript}'")

                    msg = {
                        "id": self.current_seg_id,
                        "status": "final",
                        "col1_text": transcript,
                    }
                    await self.broadcast(msg)

                    self.translate_count += len(transcript) * 2
                    log_text = f"'{transcript[:50]}'..." if len(transcript) > 50 else f"'{transcript}'"
                    print(f"Translate ({self.translate_count} chars total): {log_text}")

                    asyncio.create_task(
                        self.handle_translation(self.current_seg_id, transcript, self.target_lang_1, "trans1")
                    )
                    asyncio.create_task(
                        self.handle_translation(self.current_seg_id, transcript, self.target_lang_2, "trans2")
                    )

                self.audio_buffer = bytearray()
                self.silence_frames = 0
                self.speech_frames = 0
                self.current_seg_id = f"seg_{int(time.time()*1000)}"


# --- FastAPI / WebSocket ---
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()
engine = TranscriptionEngine(manager.broadcast)

@app.on_event("startup")
async def startup_event():
    print("--- NEW APP VERSION STARTED (Async Arch) ---")
    import asyncio
    asyncio.create_task(engine.run())
    
    # Auto-start browser
    import webbrowser
    def open_browser():
        time.sleep(1.5)
        webbrowser.open("http://127.0.0.1:8000")
    threading.Thread(target=open_browser, daemon=True).start()

@app.on_event("shutdown")
def shutdown_event():
    engine.audio_stream.stop(terminate_pyaudio=True)

@app.get("/", response_class=HTMLResponse)
async def get():
    with open(resource_path("templates/index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.get("/live", response_class=HTMLResponse)
async def get_viewer():
    with open(resource_path("templates/viewer.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "update_config":
                     engine.update_config(msg)
                elif msg.get("action") == "get_devices":
                     devices = engine.get_input_devices()
                     await websocket.send_json({"action": "device_list", "devices": devices})
                elif msg.get("action") == "shutdown":
                     print("Shutdown requested from UI")
                     engine.is_paused = True
                     engine.restart_required = True
                     engine.audio_stream.stop(terminate_pyaudio=True)
                     # Force exit after brief delay to let cleanup finish
                     def force_exit():
                         time.sleep(1)
                         print("Forcing exit...")
                         os._exit(0)
                     threading.Thread(target=force_exit, daemon=True).start()
                     import signal
                     os.kill(os.getpid(), signal.SIGINT)
            except Exception as e:
                print(f"WS Handling Error: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    import uvicorn
    import multiprocessing
    multiprocessing.freeze_support()
    print("Starting server on http://0.0.0.0:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
