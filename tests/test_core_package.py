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


if __name__ == "__main__":
    unittest.main()
