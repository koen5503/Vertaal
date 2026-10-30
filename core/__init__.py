"""Core package for Live Subtitles & Translation ('Ondertitels')."""

from core.audio import (
    CHANNELS,
    CHUNK_SIZE,
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    AudioStream,
    apply_gain,
    calculate_rms,
    resample_for_stt,
    validate_webrtc_frame,
)
try:
    from core.engine import TranscriptionEngine
except ImportError:
    TranscriptionEngine = None
from core.stt import (
    BROADCAST_CREDIT_PATTERN,
    NOISE_HALLUCINATION_PHRASES,
    TAIL_HALLUCINATION_PATTERN,
    BaseSTTEngine,
    CloudSTTEngine,
    LocalSTTEngine,
    RunPodSTTEngine,
    condense_runpod_logs,
    filter_anti_hallucination,
)
from core.translation import LANG_NAMES, META_PATTERNS, Translator, clean_translation_meta
from core.utils import (
    DEFAULT_RUNPOD_GPU,
    get_runpod_gpu_fallbacks,
    load_environment,
    parse_bool,
    resolve_pipeline_mode,
    resource_path,
)

__all__ = [
    "AudioStream",
    "TranscriptionEngine",
    "Translator",
    "BaseSTTEngine",
    "LocalSTTEngine",
    "CloudSTTEngine",
    "RunPodSTTEngine",
    "condense_runpod_logs",
    "filter_anti_hallucination",
    "NOISE_HALLUCINATION_PHRASES",
    "BROADCAST_CREDIT_PATTERN",
    "TAIL_HALLUCINATION_PATTERN",
    "FRAME_DURATION_MS",
    "SAMPLE_RATE",
    "CHANNELS",
    "CHUNK_SIZE",
    "calculate_rms",
    "apply_gain",
    "resample_for_stt",
    "validate_webrtc_frame",
    "LANG_NAMES",
    "META_PATTERNS",
    "clean_translation_meta",
    "resource_path",
    "load_environment",
    "parse_bool",
    "resolve_pipeline_mode",
    "get_runpod_gpu_fallbacks",
    "DEFAULT_RUNPOD_GPU",
]
