"""Speech-to-Text (STT) engine package."""

from core.stt.base import BaseSTTEngine
from core.stt.cloud_engine import CloudSTTEngine
from core.stt.local_engine import (
    BROADCAST_CREDIT_PATTERN,
    NOISE_HALLUCINATION_PHRASES,
    TAIL_HALLUCINATION_PATTERN,
    LocalSTTEngine,
    filter_anti_hallucination,
)
from core.stt.runpod_engine import RunPodSTTEngine, condense_runpod_logs

__all__ = [
    "BaseSTTEngine",
    "LocalSTTEngine",
    "CloudSTTEngine",
    "RunPodSTTEngine",
    "condense_runpod_logs",
    "filter_anti_hallucination",
    "NOISE_HALLUCINATION_PHRASES",
    "BROADCAST_CREDIT_PATTERN",
    "TAIL_HALLUCINATION_PATTERN",
]
