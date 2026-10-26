"""Unit tests for pipeline configuration, mode selection, and language mapping."""

import unittest
from tests.reference_logic import (
    LANG_NAMES,
    parse_bool,
    resolve_pipeline_mode,
    load_config_defaults,
    get_runpod_gpu_fallbacks,
)


class TestConfigModes(unittest.TestCase):
    """Test suite for environment variables, mode selection, and language defaults."""

    # ------------------------------------------------------------------------
    # Pipeline Mode Selection
    # ------------------------------------------------------------------------
    def test_pipeline_mode_default_is_local(self):
        """When PIPELINE_MODE is unset, default must be 'local'."""
        self.assertEqual(resolve_pipeline_mode({}), "local")

    def test_pipeline_mode_explicit_values(self):
        """Explicit valid modes ('local', 'cloud', 'runpod') must resolve cleanly."""
        modes = ["local", "cloud", "runpod"]
        for m in modes:
            env = {"PIPELINE_MODE": m}
            self.assertEqual(resolve_pipeline_mode(env), m)

    def test_pipeline_mode_case_insensitive(self):
        """Pipeline modes should be lowercased regardless of casing."""
        self.assertEqual(resolve_pipeline_mode({"PIPELINE_MODE": "CLOUD"}), "cloud")
        self.assertEqual(resolve_pipeline_mode({"PIPELINE_MODE": "RunPod"}), "runpod")
        self.assertEqual(resolve_pipeline_mode({"PIPELINE_MODE": "LoCaL"}), "local")

    # ------------------------------------------------------------------------
    # Language Mapping (LANG_NAMES)
    # ------------------------------------------------------------------------
    def test_lang_names_contains_required_languages(self):
        """LANG_NAMES must contain core languages used in church translation."""
        expected_mappings = {
            "en": "English",
            "nl": "Dutch",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "uk": "Ukrainian",
            "fa": "Farsi",
            "ar": "Arabic",
            "es": "Spanish",
            "pt": "Portuguese",
            "it": "Italian",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "tr": "Turkish",
            "pl": "Polish",
        }
        for code, name in expected_mappings.items():
            self.assertIn(code, LANG_NAMES)
            self.assertEqual(LANG_NAMES[code], name)

    def test_whisper_lang_code_extraction(self):
        """BCP-47 tags like 'nl-NL' must extract primary code 'nl' for Whisper."""
        self.assertEqual("nl-NL".split("-")[0], "nl")
        self.assertEqual("en-US".split("-")[0], "en")
        self.assertEqual("ru-RU".split("-")[0], "ru")
        self.assertEqual("de-DE".split("-")[0], "de")
        self.assertEqual("fa".split("-")[0], "fa")

    # ------------------------------------------------------------------------
    # Target Language Fallback Defaults
    # ------------------------------------------------------------------------
    def test_fallback_defaults_when_env_empty(self):
        """Default configuration values must match app.py specification."""
        cfg = load_config_defaults({})

        # Languages
        self.assertEqual(cfg["source_lang"], "nl-NL")
        self.assertEqual(cfg["target_lang_1"], "en")
        self.assertEqual(cfg["target_lang_2"], "ru")
        self.assertEqual(cfg["target_lang_3"], "de")
        self.assertEqual(cfg["target_lang_4"], "nl-NL")

        # 3rd & 4th language switches default to False
        self.assertFalse(cfg["enable_trans_3"])
        self.assertFalse(cfg["enable_trans_4"])

        # Intervals and thresholds
        self.assertEqual(cfg["translate_interval"], 10)
        self.assertEqual(cfg["silence_threshold_ms"], 300)
        self.assertEqual(cfg["max_session_minutes"], 30)
        self.assertEqual(cfg["max_duration_seconds"], 1800)
        self.assertEqual(cfg["min_local_chunk_s"], 5.0)
        self.assertEqual(cfg["interim_chunk_s"], 7.0)
        self.assertEqual(cfg["max_local_chunk_s"], 25.0)
        self.assertEqual(cfg["max_silence_flush_ms"], 2000)

    def test_boolean_env_parsing(self):
        """Test truthy vs falsy string parsing for ENABLE_TRANS_3 / 4."""
        truthy_values = ["true", "True", "TRUE", "1", "yes", "YES", True]
        for v in truthy_values:
            self.assertTrue(parse_bool(v), f"Value {v!r} should evaluate to True")

        falsy_values = ["false", "False", "0", "no", "NO", "", False]
        for v in falsy_values:
            self.assertFalse(parse_bool(v), f"Value {v!r} should evaluate to False")

    def test_environment_overrides(self):
        """Environment variables should override fallback defaults."""
        custom_env = {
            "PIPELINE_MODE": "runpod",
            "SOURCE_LANG": "en-US",
            "TARGET_LANG_1": "es",
            "TARGET_LANG_2": "zh",
            "TARGET_LANG_3": "fa",
            "TARGET_LANG_4": "uk",
            "ENABLE_TRANS_3": "true",
            "ENABLE_TRANS_4": "1",
            "TRANSLATE_INTERVAL_SEC": "5",
            "SILENCE_THRESHOLD_MS": "500",
            "MAX_SESSION_MINUTES": "45",
            "MIN_LOCAL_CHUNK_S": "8.0",
            "MAX_LOCAL_CHUNK_S": "20.0",
            "MAX_SILENCE_FLUSH_MS": "2500",
        }
        cfg = load_config_defaults(custom_env)

        self.assertEqual(cfg["pipeline_mode"], "runpod")
        self.assertEqual(cfg["source_lang"], "en-US")
        self.assertEqual(cfg["target_lang_1"], "es")
        self.assertEqual(cfg["target_lang_2"], "zh")
        self.assertEqual(cfg["target_lang_3"], "fa")
        self.assertEqual(cfg["target_lang_4"], "uk")
        self.assertTrue(cfg["enable_trans_3"])
        self.assertTrue(cfg["enable_trans_4"])
        self.assertEqual(cfg["translate_interval"], 5)
        self.assertEqual(cfg["silence_threshold_ms"], 500)
        self.assertEqual(cfg["max_session_minutes"], 45)
        self.assertEqual(cfg["max_duration_seconds"], 2700)
        self.assertEqual(cfg["min_local_chunk_s"], 8.0)
        self.assertEqual(cfg["max_local_chunk_s"], 20.0)
        self.assertEqual(cfg["max_silence_flush_ms"], 2500)

    # ------------------------------------------------------------------------
    # RunPod GPU Fallback Hierarchy
    # ------------------------------------------------------------------------
    def test_runpod_gpu_fallbacks_default_contains_all_models(self):
        """Default GPU fallback list must start with RTX 4090 and contain all models deduplicated."""
        expected_gpus = [
            "NVIDIA GeForce RTX 4090",
            "NVIDIA RTX 6000 Ada Generation",
            "NVIDIA GeForce RTX 4080",
            "NVIDIA GeForce RTX 3090",
        ]
        gpus = get_runpod_gpu_fallbacks({})
        self.assertEqual(gpus, expected_gpus)
        self.assertEqual(len(gpus), len(set(gpus)), "Fallback list must not contain duplicates")

    def test_runpod_gpu_fallbacks_custom_preferred(self):
        """When RUNPOD_GPU_ID is a supported fallback, it moves to the front with no duplicates."""
        gpus = get_runpod_gpu_fallbacks({"RUNPOD_GPU_ID": "NVIDIA GeForce RTX 4080"})
        self.assertEqual(gpus[0], "NVIDIA GeForce RTX 4080")
        self.assertEqual(len(gpus), 4)
        self.assertEqual(gpus.count("NVIDIA GeForce RTX 4080"), 1)

    def test_runpod_gpu_fallbacks_custom_nonstandard_gpu(self):
        """Custom unlisted GPU prepends to the front of all fallback models."""
        gpus = get_runpod_gpu_fallbacks({"RUNPOD_GPU_ID": "NVIDIA H100 80GB HBM3"})
        self.assertEqual(gpus[0], "NVIDIA H100 80GB HBM3")
        self.assertEqual(len(gpus), 5)
        self.assertEqual(len(gpus), len(set(gpus)))


if __name__ == "__main__":
    unittest.main()
