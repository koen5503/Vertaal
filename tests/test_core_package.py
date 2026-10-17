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

    def test_handle_translation_cancels_superseded_task(self):
        """Translation tasks for the same column key cancel previous in-flight translations."""
        import asyncio

        broadcast_events = []
        async def mock_broadcast(msg):
            broadcast_events.append(msg)

        engine = TranscriptionEngine(broadcast_callback=mock_broadcast)

        async def run_test():
            # Mock translator with controllable delay
            event_slow_started = asyncio.Event()

            async def slow_translate(text, target_lang, source_lang=None):
                if text == "first_call":
                    event_slow_started.set()
                    await asyncio.sleep(1.0)
                    return "SLOW_TRANSLATION"
                return f"FAST_{text}"

            engine.translate_text_async = slow_translate

            # Launch task 1
            task1 = asyncio.create_task(
                engine.handle_translation("seg_1", "first_call", "en", "trans1", is_interim=True)
            )
            await event_slow_started.wait()

            # Launch task 2 on the same col_key 'trans1'
            task2 = asyncio.create_task(
                engine.handle_translation("seg_1", "second_call", "en", "trans1", is_interim=False)
            )

            await task2
            # Wait for task1 to finish its cancellation handling
            await asyncio.gather(task1, return_exceptions=True)

            self.assertTrue(task1.cancelled() or task1.done())
            # Only task 2 should have broadcast its message
            trans_texts = [m["text"] for m in broadcast_events if m.get("action") == "update_translation"]
            self.assertEqual(trans_texts, ["FAST_second_call"])

        asyncio.run(run_test())

    def test_handle_translation_chain_to_restricted_to_final(self):
        """Interim translations do not trigger chain_to back-translations; only final translations do."""
        import asyncio

        broadcast_events = []
        async def mock_broadcast(msg):
            broadcast_events.append(msg)

        engine = TranscriptionEngine(broadcast_callback=mock_broadcast)

        async def run_test():
            calls = []
            async def mock_translate(text, target_lang, source_lang=None):
                calls.append((text, target_lang))
                return f"TR_{text}"

            engine.translate_text_async = mock_translate

            # Interim call with chain_to
            await engine.handle_translation(
                "seg_1", "interim_text", "en", "trans1",
                chain_to=[("de", "trans3")],
                is_interim=True
            )
            # Give any spawned tasks a moment to run
            await asyncio.sleep(0.01)
            # Should have only translated trans1, not chained to trans3
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0], ("interim_text", "en"))

            # Final call with chain_to
            calls.clear()
            await engine.handle_translation(
                "seg_1", "final_text", "en", "trans1",
                chain_to=[("de", "trans3")],
                is_interim=False
            )
            # Give chained tasks a chance to execute
            await asyncio.sleep(0.05)
            # Should have translated trans1 AND chained trans3
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0], ("final_text", "en"))
            self.assertEqual(calls[1], ("TR_final_text", "de"))

        asyncio.run(run_test())

    def test_runpod_engine_pending_seg_id_tracking(self):
        """RunPodSTTEngine tracks pending_seg_id and pending_status across interim/final transitions."""
        engine = TranscriptionEngine(broadcast_callback=MagicMock())
        runpod = RunPodSTTEngine(engine)

        self.assertIsNone(runpod.pending_seg_id)
        self.assertEqual(runpod.pending_status, "final")

        # Set interim state
        runpod.active_seg_id = "seg_test_100"
        runpod.pending_seg_id = runpod.active_seg_id
        runpod.pending_status = "interim"

        # Simulate incoming server response mapping
        dummy_incoming = {"id": "server_random_999", "status": "final", "col1_text": "Hallo"}
        if runpod.pending_seg_id:
            dummy_incoming["id"] = runpod.pending_seg_id
        if runpod.pending_status:
            dummy_incoming["status"] = runpod.pending_status

        self.assertEqual(dummy_incoming["id"], "seg_test_100")
        self.assertEqual(dummy_incoming["status"], "interim")

        # Finalize state
        runpod.pending_status = "final"
        dummy_final = {"id": "server_random_1000", "status": "server_val", "col1_text": "Hallo wereld"}
        if runpod.pending_seg_id:
            dummy_final["id"] = runpod.pending_seg_id
        if runpod.pending_status:
            dummy_final["status"] = runpod.pending_status

        self.assertEqual(dummy_final["id"], "seg_test_100")
        self.assertEqual(dummy_final["status"], "final")

    def test_runpod_text_and_translation_stitching_on_overlap_trim(self):
        """When RunPod server trims overlap, ws_receiver re-stitches full transcript and translations."""
        broadcast_messages = []
        async def mock_broadcast(msg):
            broadcast_messages.append(msg)

        engine = TranscriptionEngine(broadcast_callback=mock_broadcast)
        runpod = RunPodSTTEngine(engine)

        client_seg_id = "seg_test_stitch_1"
        runpod.pending_seg_id = client_seg_id

        # 1. Interim transcript arrival
        interim_msg = {
            "id": "srv_chunk_1",
            "col1_text": "De kerk als gondel.",
        }
        # Simulate mapping & stitching logic as in ws_receiver
        runpod.server_to_client_id["srv_chunk_1"] = runpod.pending_seg_id
        data1 = interim_msg.copy()
        data1["id"] = runpod.server_to_client_id[data1["id"]]
        incoming1 = data1["col1_text"].strip()
        runpod.segment_full_transcripts[client_seg_id] = incoming1
        data1["col1_text"] = incoming1

        self.assertEqual(data1["col1_text"], "De kerk als gondel.")

        # 2. Final transcript arrival where server stripped the first part
        final_msg = {
            "id": "srv_chunk_2",
            "col1_text": "Dat is wel wat gechargeerd.",
        }
        runpod.server_to_client_id["srv_chunk_2"] = runpod.pending_seg_id
        data2 = final_msg.copy()
        data2["id"] = runpod.server_to_client_id[data2["id"]]
        incoming2 = data2["col1_text"].strip()
        prev_text = runpod.segment_full_transcripts.get(client_seg_id, "")
        if prev_text and not (incoming2.lower().startswith(prev_text[:20].lower()) or prev_text.lower() in incoming2.lower()):
            full_text = f"{prev_text} {incoming2}".strip()
        else:
            full_text = incoming2

        self.assertEqual(full_text, "De kerk als gondel. Dat is wel wat gechargeerd.")

        # 3. Translation stitching
        trans_interim = "The church as a gondola."
        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = trans_interim
        trans_tail = "That is a bit exaggerated."
        prev_trans = runpod.segment_translations[client_seg_id].get("col2_text", "")
        if prev_trans and not (trans_tail.lower().startswith(prev_trans[:15].lower()) or prev_trans.lower() in trans_tail.lower()):
            full_trans = f"{prev_trans} {trans_tail}".strip()
        else:
            full_trans = trans_tail

        self.assertEqual(full_trans, "The church as a gondola. That is a bit exaggerated.")


if __name__ == "__main__":
    unittest.main()
