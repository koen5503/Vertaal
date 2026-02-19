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

from google.cloud import speech
from google.cloud import translate

app = FastAPI()

# --- Audio Constants ---
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000) # 480 samples


class AudioStream:
    """Captures audio from microphone using PyAudio in a background thread."""
    def __init__(self):
        self.current_device_index = None
        self._init_pyaudio()
        self.vad = webrtcvad.Vad(1) # Mode 1: Standard
        self.queue = queue.Queue()
        self.running = False
        self.thread = None

    def _init_pyaudio(self):
        self.p = pyaudio.PyAudio()
        self._open_stream()

    def _open_stream(self):
        # Close existing if open
        if hasattr(self, 'stream') and self.stream:
            self.stream.close()
            
        print(f"Opening Stream on Device Index: {self.current_device_index}")
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=CHANNELS,
            rate=SAMPLE_RATE,
            input=True,
            input_device_index=self.current_device_index,
            frames_per_buffer=CHUNK_SIZE,
        )

    def list_devices(self):
        """Re-initializes PyAudio to scan for new devices and returns list."""
        # We don't terminate self.p here to avoid breaking active stream if possible,
        # but to find NEW devices, we might need to? PyAudio usually sees them if OS sees them.
        # Let's just iterate current instance.
        devices = []
        count = self.p.get_device_count()
        for i in range(count):
            try:
                info = self.p.get_device_info_by_index(i)
                if info.get('maxInputChannels') > 0:
                    devices.append({"index": i, "name": info.get('name')})
            except Exception as e:
                print(f"Error getting device info {i}: {e}")
        return devices

    def change_device(self, index):
        """Switches input device, restarting stream if running."""
        print(f"Switching to Audio Device Index: {index}")
        self.current_device_index = index if index is not None else None
        
        was_running = self.running
        if was_running:
            self.stop(terminate_pyaudio=False)
        
        self._open_stream()
        
        if was_running:
            self.start()

    def start(self):
        if self.running: return
        self.running = True
        if self.stream.is_stopped():
            self.stream.start_stream()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()
    
    def _read_loop(self):
        """Background thread to continuously read audio."""
        print("Audio Background Thread Started")
        while self.running:
            try:
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                self.queue.put(data)
            except Exception as e:
                print(f"Audio Read Error: {e}")
                break
        print("Audio Background Thread Stopped")

    def stop(self, terminate_pyaudio=True):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if terminate_pyaudio:
            self.p.terminate()

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


class TranscriptionEngine:
    """Manages STT streaming and VAD/Stability logic."""
    def __init__(self, broadcast_callback):
        self.credentials_path = "/Users/koen/Ondertitels/ondertitels-486017-0ee48ab1ba8d.json"
        
        # Clients
        self.speech_client = speech.SpeechClient.from_service_account_json(self.credentials_path)
        self.translate_client = translate.TranslationServiceClient.from_service_account_json(self.credentials_path)
        self.project_id = "ondertitels-486017"
        self.parent = f"projects/{self.project_id}/locations/global"

        # State
        self.source_lang = "nl-NL"
        self.target_lang_1 = "en"
        self.target_lang_2 = "fr"
        self.is_paused = False
        self.restart_required = False
        
        self._setup_recognition_config()

        self.broadcast = broadcast_callback
        self.audio_stream = AudioStream()
        
        # State
        self.silence_frames = 0
        self.current_seg_id = f"seg_{int(time.time()*1000)}"
        
        # Translation Cache (simple)
        self.last_translated_text = ""

    def _setup_recognition_config(self):
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.source_lang,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
        )

    def get_input_devices(self):
        return self.audio_stream.list_devices()

    def update_config(self, config):
        print(f"Updating Config: {config}")
        
        if "paused" in config:
            self.is_paused = config["paused"]

        if "source_lang" in config and config["source_lang"] != self.source_lang:
            self.source_lang = config["source_lang"]
            self._setup_recognition_config()
            self.restart_required = True
        
        if "target_lang_1" in config:
            self.target_lang_1 = config["target_lang_1"]
            
        if "target_lang_2" in config:
            self.target_lang_2 = config["target_lang_2"]
            
        if "device_index" in config:
            idx = config["device_index"]
            # Convert to int if it's a digit string, else None (Default)
            if idx == "default" or idx is None:
                self.audio_stream.change_device(None)
            else:
                self.audio_stream.change_device(int(idx))

    async def translate_text_async(self, text, target_lang):
        """Async wrapper for Google Translate."""
        if not text or not target_lang: return ""
        if target_lang == self.source_lang.split("-")[0]: return text
        
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

    async def handle_translation(self, seg_id, text, target_lang, col_key):
        """Background task to translate and update UI."""
        if not text: return
        translated = await self.translate_text_async(text, target_lang)
        msg = {
            "action": "update_translation",
            "id": seg_id,
            "col_key": col_key, 
            "text": translated
        }
        await self.broadcast(msg)

    async def run(self):
        """Main loop."""
        print("Starting Audio Stream...")
        self.audio_stream.start()
        
        loop = asyncio.get_event_loop()
        
        while self.audio_stream.running:
            if self.is_paused:
                await asyncio.sleep(0.1)
                continue
                
            self.restart_required = False
            stop_event = asyncio.Event()
            
            # Blocking Google API call in thread
            await loop.run_in_executor(None, self._run_google_stream_sync, stop_event, loop)
            
            if self.restart_required:
                print("Restarting stream due to config change...")
            
            self.current_seg_id = f"seg_{int(time.time()*1000)}"

    def _run_google_stream_sync(self, stop_event, loop):
        """Blocking function to handle one Google Stream session."""
        
        def generator():
            chunk_count = 0
            while not stop_event.is_set():
                if self.restart_required or self.is_paused:
                    stop_event.set()
                    return

                chunk_count += 1
                # Read from Queue (Blocking)
                chunk = self.audio_stream.read_chunk()
                if not chunk: 
                    # If queue return None, stream is dead
                    stop_event.set()
                    break
                
                # Broadcast Volume
                if chunk_count % 10 == 0:
                     rms = self.audio_stream.calculate_rms(chunk)
                     vol_msg = {"type": "volume", "rms": rms}
                     asyncio.run_coroutine_threadsafe(self.broadcast(vol_msg), loop=loop)

                # VAD Check
                is_speech = self.audio_stream.vad.is_speech(chunk, SAMPLE_RATE)
                if is_speech:
                    self.silence_frames = 0
                else:
                    self.silence_frames += 1
                
                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        try:
            # Call Google API
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
                
                print(f"STT: '{transcript}' | Stability: {stability:.2f} | Final: {is_final}")
                
                silence_ms = self.silence_frames * FRAME_DURATION_MS
                
                status = "final" if is_final else "interim"
                
                # VAD CUT CHECK (Reverted to 400ms)
                if silence_ms > 400 and stability > 0.8:
                    if not is_final:
                        print(f"FORCE FINAL: '{transcript}' (Silence {silence_ms}ms)")
                        is_final = True
                        status = "final"
                        stop_event.set() 
                
                if is_final:
                    stop_event.set()

                # Basic Broadcast (Source Text)
                msg = {
                    "id": self.current_seg_id,
                    "status": status,
                    "col1_text": transcript,
                }
                asyncio.run_coroutine_threadsafe(self.broadcast(msg), loop=loop)

                # Fire Async Translations
                if is_final or stability > 0.8:
                    # Fire and forget tasks
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

@app.on_event("shutdown")
def shutdown_event():
    engine.audio_stream.stop()

@app.get("/", response_class=HTMLResponse)
async def get():
    with open("templates/index.html", "r") as f:
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
            except Exception as e:
                print(f"WS Handling Error: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)

