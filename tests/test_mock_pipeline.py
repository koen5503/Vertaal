"""Mock pipeline tests for audio accumulation, VAD silence tracking, and broadcast messaging."""

import unittest
from tests.reference_logic import (
    AudioPipelineSimulator,
    CHUNK_SIZE,
    SAMPLE_RATE,
    load_config_defaults,
)


class TestMockPipeline(unittest.TestCase):
    """Test suite for the audio accumulation and VAD dispatch pipeline."""

    def setUp(self):
        self.broadcast_messages = []
        self.transcriptions = []

        def mock_broadcast(msg):
            self.broadcast_messages.append(msg)

        def mock_transcribe(audio_bytes):
            self.transcriptions.append(audio_bytes)
            return f"Transcribed segment {len(self.transcriptions)}"

        self.mock_broadcast = mock_broadcast
        self.mock_transcribe = mock_transcribe

        self.pipeline_cfg = load_config_defaults({
            "MIN_LOCAL_CHUNK_S": "2.0",       # Shorter for fast unit tests
            "MAX_LOCAL_CHUNK_S": "5.0",
            "SILENCE_THRESHOLD_MS": "300",    # 10 frames of 30ms
            "MAX_SILENCE_FLUSH_MS": "600",    # 20 frames of 30ms
        })

        self.pipeline = AudioPipelineSimulator(
            broadcast_callback=self.mock_broadcast,
            transcribe_callback=self.mock_transcribe,
            config=self.pipeline_cfg,
        )

        # 30ms @ 16kHz 16-bit mono = 480 samples = 960 bytes
        self.dummy_speech_frame = b'\x10\x27' * CHUNK_SIZE  # amplitude ~10000
        self.dummy_silence_frame = b'\x00\x00' * CHUNK_SIZE

    def test_volume_emission_every_10_frames(self):
        """Pipeline must emit a volume event every 10 frames."""
        for i in range(25):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)

        volume_msgs = [m for m in self.broadcast_messages if m.get("type") == "volume"]
        # Frames 10 and 20 should produce volume emissions
        self.assertEqual(len(volume_msgs), 2)
        self.assertGreater(volume_msgs[0]["rms"], 0)

    def test_preroll_silence_capping(self):
        """When waiting for speech, silence buffer must be capped at 1.0 second (32,000 bytes)."""
        # Feed 100 frames of pure silence (~3.0 seconds)
        for _ in range(100):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        # Buffer length must be clamped to max 32,000 bytes (1 second pre-roll)
        self.assertLessEqual(len(self.pipeline.audio_buffer), 32000)
        self.assertEqual(self.pipeline.speech_frames, 0)
        self.assertEqual(len(self.transcriptions), 0)

    def test_vad_silence_tracking_resets_on_speech(self):
        """Silence frames counter increments on silence and resets to 0 on speech."""
        for _ in range(5):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)
        self.assertEqual(self.pipeline.silence_frames, 5)

        # Speech frame resets silence count
        self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)
        self.assertEqual(self.pipeline.silence_frames, 0)
        self.assertEqual(self.pipeline.speech_frames, 1)

    def test_silence_threshold_triggers_transcription(self):
        """When minimum duration and speech requirements are met, silence triggers STT."""
        # Feed 70 frames of speech (~2.1 seconds, meets MIN_LOCAL_CHUNK_S = 2.0s and min_speech_frames = 10)
        for _ in range(70):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)

        self.assertEqual(len(self.transcriptions), 0)

        # Feed silence until silence_threshold_ms (300ms = 10 frames)
        for _ in range(10):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        # Should have triggered transcription and broadcast final segment
        self.assertEqual(len(self.transcriptions), 1)
        final_msgs = [m for m in self.broadcast_messages if m.get("status") == "final"]
        self.assertEqual(len(final_msgs), 1)
        self.assertEqual(final_msgs[0]["col1_text"], "Transcribed segment 1")

        # Pipeline state should be reset
        self.assertEqual(len(self.pipeline.audio_buffer), 0)
        self.assertEqual(self.pipeline.speech_frames, 0)
        self.assertEqual(self.pipeline.silence_frames, 0)

    def test_hard_cap_forces_transcription_without_silence(self):
        """Buffer reaching MAX_LOCAL_CHUNK_S (5.0s) forces transcription even with continuous speech."""
        # 5.0s @ 30ms/frame = ~167 frames
        for _ in range(170):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)

        # Should have forced transcription due to hard cap
        self.assertGreaterEqual(len(self.transcriptions), 1)
        final_msgs = [m for m in self.broadcast_messages if m.get("status") == "final"]
        self.assertGreaterEqual(len(final_msgs), 1)

    def test_force_flush_on_long_silence(self):
        """If audio duration < min_local_chunk_s, but silence reaches max_silence_flush_ms, force flush."""
        # Feed 15 speech frames (~450ms, meets min_speech >= 10, but < 2.0s min_local_chunk_s)
        for _ in range(15):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)

        self.assertEqual(len(self.transcriptions), 0)

        # Feed silence up to normal silence threshold (10 frames = 300ms)
        for _ in range(10):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)
        # Not yet flushed because buffer < 2.0s
        self.assertEqual(len(self.transcriptions), 0)

        # Continue silence until max_silence_flush_ms (600ms = 20 frames total)
        for _ in range(11):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        # Now force_flush must trigger transcription
        self.assertEqual(len(self.transcriptions), 1)

    def test_mic_clicks_under_3_frames_discarded(self):
        """Isolated mic clicks (< 3 speech frames) flushed by long silence are discarded."""
        # Feed only 2 speech frames (60ms mic bump / click)
        self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)
        self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)

        # Feed silence until force_flush (20 frames = 600ms)
        for _ in range(20):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        # Mic click should be discarded without running transcription
        self.assertEqual(len(self.transcriptions), 0)
        self.assertEqual(len(self.pipeline.audio_buffer), 0)
        self.assertEqual(self.pipeline.speech_frames, 0)

    def test_sequential_utterances(self):
        """Sequential speech blocks generate sequential segments cleanly."""
        # Utterance 1
        for _ in range(70):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)
        for _ in range(10):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        self.assertEqual(len(self.transcriptions), 1)

        # Utterance 2
        for _ in range(70):
            self.pipeline.feed_frame(self.dummy_speech_frame, is_speech=True)
        for _ in range(10):
            self.pipeline.feed_frame(self.dummy_silence_frame, is_speech=False)

        self.assertEqual(len(self.transcriptions), 2)
        final_msgs = [m for m in self.broadcast_messages if m.get("status") == "final"]
        self.assertEqual(len(final_msgs), 2)
        self.assertEqual(final_msgs[0]["col1_text"], "Transcribed segment 1")
        self.assertEqual(final_msgs[1]["col1_text"], "Transcribed segment 2")


if __name__ == "__main__":
    unittest.main()
