"""Reference logic and pure Python implementations extracted directly from app.py.

These implementations replicate the exact behavior in app.py without
requiring external hardware (microphones, sound cards), external cloud services,
or heavy dependencies (pyaudio, sounddevice, google-cloud-speech) so that unit
and regression tests can run in any environment with standard Python.
"""

import math
import struct
import os
import string
from typing import Optional, List, Dict, Any, Callable


# ============================================================================
# Audio Constants (matching app.py lines 76-80)
# ============================================================================
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
CHANNELS = 1
CHUNK_SIZE = int(SAMPLE_RATE * FRAME_DURATION_MS / 1000)  # 480 samples = 960 bytes (16-bit mono)


# ============================================================================
# Audio Processing Utilities (matching app.py lines 307-345 and 507-524)
# ============================================================================
def calculate_rms(data: bytes) -> int:
    """Calculate Root Mean Square (RMS) volume of 16-bit linear PCM audio.
    
    Directly mirrors AudioStream.calculate_rms in app.py line 507.
    """
    count = len(data) // 2
    if count == 0:
        return 0
    fmt = "<%dh" % count
    shorts = struct.unpack(fmt, data)
    sum_squares = sum(s * s for s in shorts)
    return int(math.sqrt(sum_squares / count))


def apply_gain(data: bytes, gain: float) -> bytes:
    """Apply linear gain multiplier with 16-bit signed integer clipping.
    
    Directly mirrors AudioStream.apply_gain in app.py line 514.
    """
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


def resample_for_stt(data: bytes, src_rate: int, src_channels: int, src_width: int, target_rate: int = 16000) -> bytes:
    """Convert audio data to 16kHz mono 16-bit LINEAR16 for STT.
    
    Directly mirrors AudioStream._resample_for_stt in app.py line 307.
    """
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


def validate_webrtc_frame(data: bytes, sample_rate: int = 16000) -> bool:
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
# Anti-Hallucination Filter (matching app.py lines 1354-1431)
# ============================================================================
def filter_anti_hallucination(
    text: Optional[str],
    last_transcript: str = "",
    glossary_phrases: Optional[List[str]] = None,
) -> Optional[str]:
    """Run anti-hallucination filters on Whisper transcription output.
    
    Directly mirrors the filter block in TranscriptionEngine._transcribe_buffer
    (app.py lines 1354-1431).
    """
    if not text:
        return None

    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    words = clean_text.split()

    if not words:
        return None

    # Prompt Leakage Filter (Whisper Hallucination)
    last_trans = last_transcript or ""
    if last_trans:
        clean_last = last_trans.translate(str.maketrans('', '', string.punctuation)).lower()
        last_words = clean_last.split()

        # a) Exact Prompt Leakage
        if clean_text == clean_last:
            return None

        # b) Partial Prompt Leakage (len >= 3 and clean_text in clean_last)
        if len(words) >= 3 and clean_text in clean_last:
            return None

        # c) Overlapping sliding-window prefix trimmer
        max_overlap = min(len(last_words), len(words))
        best_overlap = 0
        for i in range(max_overlap, 0, -1):
            if last_words[-i:] == words[:i]:
                best_overlap = i
                break

        if best_overlap > 0:
            original_words = text.split()
            text = " ".join(original_words[best_overlap:])
            if not text:
                return None

            clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
            words = clean_text.split()
            if not words:
                return None

    # 1. Repetition filter
    # A. Exactly the same word repeated
    if len(words) >= 2 and len(set(words)) == 1:
        return None

    # B. Consecutive repeating words OR extremely low unique word count
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    # d) Consecutive repeating word loops (>= 4)
    if max_consecutive >= 4:
        return None

    # e) Extremely low unique word count (< 30% for >= 5 words)
    if len(words) >= 5 and len(set(words)) / len(words) < 0.3:
        return None

    # f) Isolated glossary word filter (len <= 2 on breath/noise)
    if glossary_phrases and len(words) <= 2:
        glos_lower = [p.lower() for p in glossary_phrases]
        if any(w in glos_lower for w in words) or clean_text in glos_lower:
            return None

    return text if text else None


# ============================================================================
# Language Mapping & Configuration (matching app.py lines 527-567, 634-660)
# ============================================================================
LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "ru": "Russian",
    "uk": "Ukrainian", "fa": "Farsi", "ar": "Arabic", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "pl": "Polish",
}


def parse_bool(val: Any) -> bool:
    """Parse string or boolean to boolean value."""
    if isinstance(val, bool):
        return val
    return str(val).lower() in ("true", "1", "yes")


def resolve_pipeline_mode(env: Optional[Dict[str, str]] = None) -> str:
    """Resolve pipeline mode from environment dictionary."""
    if env is None:
        env = os.environ
    return env.get("PIPELINE_MODE", "local").lower()


def load_config_defaults(env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Load default configuration from environment or fallback defaults."""
    if env is None:
        env = os.environ

    return {
        "pipeline_mode": env.get("PIPELINE_MODE", "local").lower(),
        "source_lang": env.get("SOURCE_LANG", "nl-NL"),
        "target_lang_1": env.get("TARGET_LANG_1", "en"),
        "target_lang_2": env.get("TARGET_LANG_2", "ru"),
        "target_lang_3": env.get("TARGET_LANG_3", "de"),
        "target_lang_4": env.get("TARGET_LANG_4", "nl-NL"),
        "enable_trans_3": parse_bool(env.get("ENABLE_TRANS_3", "false")),
        "enable_trans_4": parse_bool(env.get("ENABLE_TRANS_4", "false")),
        "translate_interval": int(env.get("TRANSLATE_INTERVAL_SEC", "10")),
        "silence_threshold_ms": int(env.get("SILENCE_THRESHOLD_MS", "400")),
        "max_session_minutes": int(env.get("MAX_SESSION_MINUTES", "30")),
        "max_duration_seconds": int(env.get("MAX_SESSION_MINUTES", "30")) * 60,
        "min_local_chunk_s": float(env.get("MIN_LOCAL_CHUNK_S", "10.0")),
        "max_local_chunk_s": float(env.get("MAX_LOCAL_CHUNK_S", "25.0")),
        "max_silence_flush_ms": int(env.get("MAX_SILENCE_FLUSH_MS", "2000")),
        "min_speech_frames": 10,
    }


def get_runpod_gpu_fallbacks(env: Optional[Dict[str, str]] = None) -> List[str]:
    """Build the deduplicated priority list of GPUs for RunPod deployment.
    
    Directly mirrors TranscriptionEngine._deploy_runpod_pod in app.py.
    """
    if env is None:
        env = os.environ
    preferred_gpu = env.get("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")
    candidate_gpus = [
        preferred_gpu,
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 3090",
        "NVIDIA RTX A5000",
        "NVIDIA RTX A4000",
        "NVIDIA RTX A4500",
        "NVIDIA GeForce RTX 4080",
        "NVIDIA L4",
        "NVIDIA RTX 6000 Ada Generation",
    ]
    return list(dict.fromkeys(candidate_gpus))


# ============================================================================
# Mock Audio Pipeline Simulator (matching app.py lines 1468-1561)
# ============================================================================
class AudioPipelineSimulator:
    """Simulates the audio accumulation, VAD tracking, and dispatch loop
    from TranscriptionEngine._run_local (app.py lines 1468-1561).
    """

    def __init__(
        self,
        broadcast_callback: Optional[Callable[[Dict[str, Any]], Any]] = None,
        transcribe_callback: Optional[Callable[[bytes], Optional[str]]] = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        self.broadcast_callback = broadcast_callback
        self.transcribe_callback = transcribe_callback
        cfg = config or load_config_defaults({})

        self.silence_threshold_ms = cfg["silence_threshold_ms"]
        self.max_silence_flush_ms = cfg["max_silence_flush_ms"]
        self.min_local_chunk_s = cfg["min_local_chunk_s"]
        self.max_local_chunk_s = cfg["max_local_chunk_s"]
        self.min_speech_frames = cfg.get("min_speech_frames", 10)

        # State
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.chunk_count = 0
        self.current_seg_id = "seg_test_1"
        self.last_transcript = ""
        self.emitted_messages: List[Dict[str, Any]] = []

    def feed_frame(self, chunk: bytes, is_speech: bool) -> Optional[str]:
        """Process a single 30ms chunk of audio."""
        self.chunk_count += 1

        # Periodic volume emission every 10 frames
        if self.chunk_count % 10 == 0:
            rms = calculate_rms(chunk)
            msg = {"type": "volume", "rms": rms}
            self.emitted_messages.append(msg)
            if self.broadcast_callback:
                self.broadcast_callback(msg)

        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
        else:
            self.silence_frames += 1

        self.audio_buffer.extend(chunk)

        # Keep max 1.0s pre-roll silence while waiting for speech
        if self.speech_frames == 0:
            pre_roll_bytes = 32000  # 1 sec @ 16kHz 16-bit mono
            if len(self.audio_buffer) > pre_roll_bytes:
                self.audio_buffer = bytearray(self.audio_buffer[-pre_roll_bytes:])

        silence_ms = self.silence_frames * FRAME_DURATION_MS
        buffer_duration_s = len(self.audio_buffer) / (SAMPLE_RATE * 2)

        hard_cap = buffer_duration_s >= self.max_local_chunk_s
        force_flush = silence_ms >= self.max_silence_flush_ms

        if (hard_cap or force_flush or silence_ms >= self.silence_threshold_ms) and self.speech_frames > 0:
            # Discard trapped pure mic clicks (under 90ms / 3 frames)
            if force_flush and self.speech_frames < 3:
                self.audio_buffer = bytearray()
                self.speech_frames = 0
                self.silence_frames = 0
                return None

            met_speech_req = self.speech_frames >= self.min_speech_frames or ((force_flush or hard_cap) and self.speech_frames >= 3)
            if not met_speech_req:
                return None

            if buffer_duration_s < self.min_local_chunk_s and not force_flush and not hard_cap:
                # Do not reset silence frames, let them increment
                return None

            # Transcription trigger
            transcript = None
            if self.transcribe_callback:
                transcript = self.transcribe_callback(bytes(self.audio_buffer))

            if transcript:
                self.last_transcript = transcript[-200:]
                msg = {
                    "id": self.current_seg_id,
                    "status": "final",
                    "col1_text": transcript,
                }
                self.emitted_messages.append(msg)
                if self.broadcast_callback:
                    self.broadcast_callback(msg)

            # Reset segment buffer and counters
            self.audio_buffer = bytearray()
            self.silence_frames = 0
            self.speech_frames = 0
            self.current_seg_id = f"seg_test_{self.chunk_count}"
            return transcript

        return None
