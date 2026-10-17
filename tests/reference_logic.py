"""Reference logic and pure Python implementations extracted directly from app.py / core.

These implementations replicate the exact behavior in app.py without
requiring external hardware (microphones, sound cards), external cloud services,
or heavy dependencies (pyaudio, sounddevice, google-cloud-speech) so that unit
and regression tests can run in any environment with standard Python.
"""

import os
from typing import Any, Callable, Dict, List, Optional

from core.audio import (
    CHANNELS,
    CHUNK_SIZE,
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    apply_gain,
    calculate_rms,
    resample_for_stt,
    validate_webrtc_frame,
)
from core.stt import filter_anti_hallucination
from core.translation import LANG_NAMES
from core.utils import (
    get_runpod_gpu_fallbacks,
    parse_bool,
    resolve_pipeline_mode,
)


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


# ============================================================================
# Mock Audio Pipeline Simulator (matching app.py / core.stt.local_engine)
# ============================================================================
class AudioPipelineSimulator:
    """Simulates the audio accumulation, VAD tracking, and dispatch loop."""

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
