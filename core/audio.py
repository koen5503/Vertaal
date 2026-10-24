"""Audio capturing, stream management, resampling, and VAD processing."""

import glob
import math
import os
import queue
import socket
import struct
import threading
import time
import wave
from typing import Any, Dict, List, Optional

from core.utils import resource_path

try:
    import webrtcvad
except ImportError:
    webrtcvad = None

try:
    import pyaudio
except ImportError:
    pyaudio = None

try:
    import sounddevice as sd
except ImportError:
    sd = None

# --- Audio Constants ---
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples = 960 bytes (16-bit mono)


# ============================================================================
# Audio Processing Utilities
# ============================================================================
def calculate_rms(data: bytes) -> int:
    """Calculate Root Mean Square (RMS) volume of 16-bit linear PCM audio."""
    count = len(data) // 2
    if count == 0:
        return 0
    fmt = "<%dh" % count
    shorts = struct.unpack(fmt, data)
    sum_squares = sum(s * s for s in shorts)
    return int(math.sqrt(sum_squares / count))


def apply_gain(data: bytes, gain: float) -> bytes:
    """Apply linear gain multiplier with 16-bit signed integer clipping."""
    count = len(data) // 2
    shorts = struct.unpack(f"<{count}h", data)
    scaled = []
    for s in shorts:
        val = int(s * gain)
        if val > 32767:
            val = 32767
        if val < -32768:
            val = -32768
        scaled.append(val)
    return struct.pack(f"<{count}h", *scaled)


def resample_for_stt(data: bytes, src_rate: int, src_channels: int, src_width: int, target_rate: int = SAMPLE_RATE) -> bytes:
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
            mono.append(sum(samples[i:i + src_channels]) // src_channels)
        samples = mono

    # Resample to target_rate using linear interpolation
    if src_rate != target_rate:
        ratio = src_rate / target_rate
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


def validate_webrtc_frame(data: bytes, sample_rate: int = SAMPLE_RATE) -> bool:
    """Verify that chunk satisfies WebRTC VAD constraints:
    - Sample rate must be 8000, 16000, 32000, or 48000 Hz.
    - Duration must be exactly 10, 20, or 30 ms.
    - Format is 16-bit mono linear PCM (2 bytes per sample).
    """
    valid_rates = {8000, 16000, 32000, 48000}
    if sample_rate not in valid_rates:
        return False

    valid_durations_ms = {10, 20, 30}
    bytes_per_sample = 2
    num_samples = len(data) // bytes_per_sample

    if len(data) % bytes_per_sample != 0:
        return False

    duration_ms = (num_samples * 1000) / sample_rate
    return duration_ms in valid_durations_ms


# ============================================================================
# AudioStream
# ============================================================================
class AudioStream:
    """Captures audio from microphone or file in a background thread."""

    def __init__(self):
        self.p = pyaudio.PyAudio() if pyaudio else None
        self.vad = webrtcvad.Vad(1) if webrtcvad else None  # Mode 1: Standard
        self.queue = queue.Queue()
        self.running = False
        self.playback_paused = False
        self.thread = None
        self.sd_stream = None  # For sounddevice (Windows/ASIO)
        pipeline_mode = os.getenv("PIPELINE_MODE", "local").lower()
        self.is_connected = (pipeline_mode != "runpod")

        # Default to VoiceMeeter if available, else demo file
        self.current_device_index = "demo_file"
        self.stream = None

        # Auto-discovery of VoiceMeeter or other preferred device
        preferred = os.getenv("DEFAULT_DEVICE_NAME", "Voicemeeter Out B1")
        devs = self.list_devices()

        # Debug: Print all devices to terminal on startup
        print("--- Detected Audio Devices ---")
        for d in devs:
            print(f"  [{d['index']}] {d['name']}")
        print("-------------------------------")

        # First pass: try to find an EXACT match (ignoring case and "Mic:" prefix)
        for d in devs:
            clean_name = d['name'].lower().replace("mic:", "").strip()
            if preferred.lower() == clean_name or preferred.lower() in clean_name:
                # If we find B1, we stop immediately
                if "b1" in clean_name and "b1" in preferred.lower():
                    self.current_device_index = d['index']
                    print(f"Startup: Auto-selected device '{d['name']}' (Index: {d['index']})")
                    break

        if self.current_device_index == "demo_file":
            # Second pass: loose match if B1 specifically wasn't found
            for d in devs:
                if preferred.lower() in d['name'].lower():
                    self.current_device_index = d['index']
                    print(f"Startup: Auto-selected device (Loose Match) '{d['name']}' (Index: {d['index']})")
                    break

        if self.current_device_index == "demo_file":
            print(f"Startup mode: Demo File ({os.getenv('DEMO_FILE_PATH', 'BV.wav')})")

        self._open_stream()

    def is_speech(self, chunk: bytes) -> bool:
        """Unified VAD check. Uses WebRTC VAD (identical to Mac)."""
        if not self.vad:
            return False
        try:
            return self.vad.is_speech(chunk, SAMPLE_RATE)
        except Exception:
            return False

    def _open_stream(self):
        # Close existing if open
        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.close()
            except Exception:
                pass
            self.stream = None

        if (
            self.current_device_index == "demo_file"
            or str(self.current_device_index).startswith("file_")
            or self.current_device_index == "vban"
            or str(self.current_device_index).startswith("sd_")
        ):
            print(f"Preparing to stream from Virtual/SoundDevice Endpoint: {self.current_device_index}")
            self.stream = None
            return

        if not pyaudio or not self.p:
            print("PyAudio not available. Skipping PyAudio stream open.")
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

    def list_devices(self) -> List[Dict[str, Any]]:
        """Uses existing PyAudio and SoundDevice instances to scan for devices and returns list."""
        devices = [{"index": "demo_file", "name": "Demo Loop (Bert.wav)"}]

        # 1. PyAudio (Mac / standard)
        if pyaudio and self.p:
            try:
                for i in range(self.p.get_device_count()):
                    info = self.p.get_device_info_by_index(i)
                    if info.get('maxInputChannels', 0) > 0:
                        devices.append({"index": i, "name": f"Mic: {info.get('name')}"})
            except Exception as e:
                print(f"Error getting PyAudio devices: {e}")

        # 2. SoundDevice (Windows / ASIO / Pro Audio)
        if sd:
            try:
                sd_devs = sd.query_devices()
                for i, dev in enumerate(sd_devs):
                    if dev['max_input_channels'] > 0:
                        host_api = sd.query_hostapis(dev['hostapi'])['name']
                        devices.append({"index": f"sd_{i}", "name": f"SD: {dev['name']} ({host_api})"})
            except Exception as e:
                print(f"Error getting SoundDevice devices: {e}")

        # 3. Virtual files & VBAN
        wav_files = set(os.path.basename(f) for f in glob.glob("*.wav"))
        for w in sorted(wav_files):
            devices.append({"index": f"file_{w}", "name": f"File: {w}"})

        devices.append({"index": "vban", "name": "Direct Network (VBAN)"})
        return devices

    def change_device(self, index: Any):
        """Switches input device by thoroughly resetting the stream."""
        print(f"Switching to Audio Device Index: {index}")
        normalized_index = index if index is not None else None

        # Allow re-selection of demo_file (actual file may have changed)
        if (
            self.current_device_index == normalized_index
            and not str(normalized_index).startswith("file_")
            and normalized_index not in ("demo_file", "vban")
        ):
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
        if pyaudio and self.p:
            try:
                self.p.terminate()
            except Exception:
                pass
            self.p = pyaudio.PyAudio()

        # Reopen input stream on fresh instance
        self._open_stream()

    def start(self):
        if self.running:
            return
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
        if self.current_device_index == "demo_file" or str(self.current_device_index).startswith("file_"):
            self._read_file_loop()
            return
        elif self.current_device_index == "vban":
            self._read_vban_loop()
            return
        elif str(self.current_device_index).startswith("sd_"):
            self._read_sounddevice_loop()
            return

        while self.running:
            try:
                if self.stream is None:
                    time.sleep(0.5)
                    continue
                data = self.stream.read(CHUNK_SIZE, exception_on_overflow=False)
                if not self.is_connected:
                    continue
                self.queue.put(data)
            except Exception as e:
                print(f"Audio Read Error: {e}")
                break

    def _read_sounddevice_loop(self):
        """Background thread for sounddevice capture (Windows ASIO)."""
        import numpy as np
        try:
            dev_idx = int(str(self.current_device_index).split("_")[1])
            print(f"Starting SoundDevice loop on index {dev_idx} (16kHz Mono)")

            # WebRTC VAD needs exactly 10, 20 or 30ms.
            # At 16000Hz, 30ms is 480 frames.
            VAD_FRAMES = 480

            def callback(indata, frames, ctime, status):
                if status:
                    print(f"SD Status: {status}")
                if not self.is_connected:
                    return
                # indata is float32 [-1.0, 1.0], convert to int16
                pcm = (indata[:, 0] * 32767).astype(np.int16).tobytes()
                self.queue.put(pcm)

            with sd.InputStream(
                device=dev_idx,
                channels=CHANNELS,
                samplerate=SAMPLE_RATE,
                dtype='float32',
                blocksize=VAD_FRAMES,
                callback=callback
            ):
                while self.running:
                    time.sleep(0.1)
        except Exception as e:
            print(f"SoundDevice Stream Error: {e}")

    def _resample_for_stt(self, data, src_rate, src_channels, src_width):
        """Convert audio data to 16kHz mono 16-bit LINEAR16 for STT."""
        return resample_for_stt(data, src_rate, src_channels, src_width, target_rate=SAMPLE_RATE)

    def _read_vban_loop(self):
        """Background thread to natively listen to VBAN UDP Audio from Windows."""
        vban_target = os.getenv("VBAN_STREAM_NAME", "ringleiding").lower()

        SR_LIST = [
            6000, 12000, 24000, 48000, 96000, 192000, 384000,
            8000, 16000, 32000, 64000, 128000, 256000, 512000,
            11025, 22050, 44100, 88200, 176400, 352800,
            7110, 14220, 28440, 56880, 113760, 227520
        ]

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("0.0.0.0", 6981))
        sock.settimeout(0.5)

        print(f"VBAN Listener started on UDP 6981. Waiting for stream: '{vban_target}'")

        vban_pcm_accumulator = bytearray()

        while self.running:
            try:
                data, addr = sock.recvfrom(2048)

                if len(data) <= 28:
                    continue

                vban_header = data[:28]
                tag, sp_sr, sams, chans, fmt, stream_name, frame = struct.unpack("<4sBBBB16sI", vban_header)

                if tag != b'VBAN':
                    continue

                stream_name_str = stream_name.decode('ascii', 'replace').strip('\x00').strip().lower()
                if stream_name_str != vban_target:
                    continue

                sr_index = sp_sr & 0x1F
                sample_rate = SR_LIST[sr_index] if sr_index < len(SR_LIST) else 48000
                nb_channels = chans + 1

                payload = data[28:]

                # Push uncompressed VBAN UDP chunk straight to resampler -> audio Queue
                resampled = self._resample_for_stt(payload, sample_rate, nb_channels, 2)

                if resampled:
                    vban_pcm_accumulator.extend(resampled)
                    # WebRTC VAD requires exact 10, 20 or 30ms.
                    # At 16kHz 16-bit mono: 30ms = 480 frames = 960 bytes.
                    TARGET_BYTES = 960
                    while len(vban_pcm_accumulator) >= TARGET_BYTES:
                        exact_chunk = bytes(vban_pcm_accumulator[:TARGET_BYTES])
                        vban_pcm_accumulator = vban_pcm_accumulator[TARGET_BYTES:]
                        self.queue.put(exact_chunk)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"VBAN read error: {e}")
                time.sleep(1)

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

            sleep_time = CHUNK_SIZE / src_rate

            # In RunPod mode, gate demo file playback at 0:00 until WebSocket is connected
            if not self.is_connected:
                print(f"[Audio] RunPod connection pending: gating demo playback of '{demo_name}' at 0:00...")
                while self.running and not self.is_connected:
                    time.sleep(0.05)
                if not self.running:
                    return
                print(f"[Audio] RunPod connected! Commencing playback of '{demo_name}' from 0:00.")

            out_stream = None
            if self.p and pyaudio:
                try:
                    out_stream = self.p.open(
                        format=self.p.get_format_from_width(src_width),
                        channels=src_channels,
                        rate=src_rate,
                        output=True
                    )
                except Exception as e:
                    print(f"Warning: Failed to open PyAudio speaker out stream: {e}")
                    out_stream = None

            data = wf.readframes(CHUNK_SIZE)

            while self.running:
                if len(data) == 0:
                    print(f"Looping demo file: {demo_name}")
                    wf.rewind()
                    data = wf.readframes(CHUNK_SIZE)
                    if len(data) == 0:
                        break

                if not self.is_connected or self.playback_paused:
                    time.sleep(0.05)
                    continue

                start_t = time.time()
                if out_stream:
                    try:
                        out_stream.write(data)
                    except Exception as e:
                        print(f"Warning: Write to PyAudio out failed: {e}")
                        try:
                            out_stream.close()
                        except Exception:
                            pass
                        out_stream = None

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

            if out_stream:
                try:
                    out_stream.stop_stream()
                    out_stream.close()
                except Exception:
                    pass
            wf.close()
        except Exception as e:
            print(f"File Read Error: {e}")

        print("File Demo Finished or Stopped")
        self.running = False

    def stop(self, terminate_pyaudio: bool = False):
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        if hasattr(self, 'stream') and self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception:
                pass

        if terminate_pyaudio and hasattr(self, 'p') and self.p:
            try:
                self.p.terminate()
            except Exception:
                pass

    def read_chunk(self) -> bytes:
        """Blocking get from queue."""
        try:
            return self.queue.get(timeout=1.0)
        except queue.Empty:
            print("WARNING: Audio Queue Empty (generating silence)")
            return b'\x00' * (CHUNK_SIZE * 2)

    def calculate_rms(self, data: bytes) -> int:
        return calculate_rms(data)

    def apply_gain(self, data: bytes, gain: float) -> bytes:
        return apply_gain(data, gain)
