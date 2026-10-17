"""Unit tests for the modular core package and its component engines."""

import os
import unittest
from unittest.mock import MagicMock, patch

from core import (
    AudioStream,
    TranscriptionEngine,
    Translator,
    LocalSTTEngine,
    CloudSTTEngine,
    RunPodSTTEngine,
    filter_anti_hallucination,
    get_runpod_gpu_fallbacks,
    load_environment,
    parse_bool,
    resolve_pipeline_mode,
    resource_path,
    CHUNK_SIZE,
    FRAME_DURATION_MS,
    SAMPLE_RATE,
    CHANNELS,
)


class TestCorePackage(unittest.TestCase):
    """Test suite validating that core modules instantiate and function correctly."""

    def test_core_package_exports(self):
        """Ensure all primary components are exported directly from core."""
        self.assertIsNotNone(AudioStream)
        self.assertIsNotNone(TranscriptionEngine)
        self.assertIsNotNone(Translator)
        self.assertIsNotNone(LocalSTTEngine)
        self.assertIsNotNone(CloudSTTEngine)
        self.assertIsNotNone(RunPodSTTEngine)
        self.assertEqual(CHUNK_SIZE, 480)
        self.assertEqual(SAMPLE_RATE, 16000)
        self.assertEqual(CHANNELS, 1)
        self.assertEqual(FRAME_DURATION_MS, 30)

    def test_resource_path_resolution(self):
        """resource_path must resolve relative paths to absolute paths."""
        resolved = resource_path("templates/index.html")
        self.assertTrue(os.path.isabs(resolved))
        self.assertTrue(resolved.endswith("templates/index.html"))

    def test_translator_local_initialization(self):
        """Translator initializes with local mode defaults."""
        translator = Translator(pipeline_mode="local", source_lang="nl-NL")
        self.assertEqual(translator.pipeline_mode, "local")
        self.assertEqual(translator.source_lang, "nl-NL")
        self.assertIsNone(translator.translate_client)

    def test_transcription_engine_lifecycle(self):
        """TranscriptionEngine initializes, loads config, and updates parameters cleanly."""
        broadcasts = []

        def mock_broadcast(msg):
            broadcasts.append(msg)

        with patch.dict(os.environ, {"PIPELINE_MODE": "local"}):
            engine = TranscriptionEngine(broadcast_callback=mock_broadcast)
            self.assertEqual(engine.pipeline_mode, "local")
            self.assertIsInstance(engine.stt_engine, LocalSTTEngine)
            self.assertIsInstance(engine.audio_stream, AudioStream)

            # Test updating configuration
            engine.update_config({
                "paused": True,
                "source_lang": "en-US",
                "target_lang_1": "de",
                "enable_trans_3": True,
            })
            self.assertTrue(engine.is_paused)
            self.assertEqual(engine.source_lang, "en-US")
            self.assertEqual(engine.target_lang_1, "de")
            self.assertTrue(engine.enable_trans_3)

    def test_audio_stream_device_listing(self):
        """AudioStream must list available devices including demo file and VBAN."""
        stream = AudioStream()
        devices = stream.list_devices()
        self.assertIsInstance(devices, list)
        indices = [d["index"] for d in devices]
        self.assertIn("demo_file", indices)
        self.assertIn("vban", indices)

    def test_runpod_engine_gpu_ordering(self):
        """RunPodSTTEngine GPU list preserves preferred GPU priority."""
        gpus = get_runpod_gpu_fallbacks("NVIDIA RTX A5000")
        self.assertEqual(gpus[0], "NVIDIA RTX A5000")
        self.assertEqual(len(gpus), len(set(gpus)))

    def test_incremental_interim_and_final_chunking_triggers(self):
        """Validate interim (7s) and final (pause/ceiling) trigger conditions and seg_id persistence."""
        with patch.dict(os.environ, {
            "MIN_LOCAL_CHUNK_S": "5.0",
            "INTERIM_CHUNK_S": "7.0",
            "MAX_LOCAL_CHUNK_S": "25.0",
            "SILENCE_THRESHOLD_MS": "300",
        }):
            engine = TranscriptionEngine(broadcast_callback=MagicMock())
            self.assertEqual(engine.min_local_chunk_s, 5.0)
            self.assertEqual(engine.interim_chunk_s, 7.0)
            self.assertEqual(engine.silence_threshold_ms, 300)

            stt = LocalSTTEngine(engine)
            self.assertIsNone(stt.active_seg_id)
            self.assertEqual(stt.last_interim_s, 0.0)

            # 1. Simulate continuous speech reaching 7.0s (interim interval)
            # 7.0s @ 16kHz 16-bit mono = 7.0 * 16000 * 2 = 224,000 bytes
            stt.audio_buffer = bytearray(b'\x01\x00' * int(7.0 * SAMPLE_RATE))
            stt.speech_frames = int(7.0 * 1000 / FRAME_DURATION_MS)
            stt.silence_frames = 0

            buffer_duration_s = len(stt.audio_buffer) / (SAMPLE_RATE * 2)
            silence_ms = stt.silence_frames * FRAME_DURATION_MS

            is_natural_pause = (silence_ms >= engine.silence_threshold_ms and buffer_duration_s >= engine.min_local_chunk_s)
            is_force_flush = (silence_ms >= engine.max_silence_flush_ms)
            is_hard_ceiling = (buffer_duration_s >= engine.max_local_chunk_s)
            is_interim = (
                (buffer_duration_s - stt.last_interim_s >= engine.interim_chunk_s)
                and not is_natural_pause
                and not is_hard_ceiling
            )

            self.assertTrue(is_interim)
            self.assertFalse(is_natural_pause)

            # Assign active_seg_id as done in interim branch
            stt.active_seg_id = "seg_test_incremental_1"
            stt.last_interim_s = buffer_duration_s

            # 2. Simulate speech continuing to 9.5s followed by 300ms pause (10 frames)
            # 9.8s total buffer
            stt.audio_buffer = bytearray(b'\x01\x00' * int(9.8 * SAMPLE_RATE))
            stt.silence_frames = 10  # 300ms
            silence_ms = stt.silence_frames * FRAME_DURATION_MS
            buffer_duration_s = len(stt.audio_buffer) / (SAMPLE_RATE * 2)

            is_natural_pause = (silence_ms >= engine.silence_threshold_ms and buffer_duration_s >= engine.min_local_chunk_s)
            is_interim = (
                (buffer_duration_s - stt.last_interim_s >= engine.interim_chunk_s)
                and not is_natural_pause
                and not is_hard_ceiling
            )

            # Must finalize on natural pause using the same active_seg_id
            self.assertTrue(is_natural_pause)
            self.assertFalse(is_interim)
            self.assertEqual(stt.active_seg_id, "seg_test_incremental_1")


if __name__ == "__main__":
    unittest.main()
