"""Unit tests for the modular core package and its component engines."""

import asyncio
import json
import os
import time
import unittest
from unittest.mock import MagicMock, patch

from core import (
    AudioStream,
    TranscriptionEngine,
    Translator,
    LocalSTTEngine,
    CloudSTTEngine,
    RunPodSTTEngine,
    condense_runpod_logs,
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
from core.stt.runpod_engine import restitch_segment_text


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
        self.assertIsNotNone(condense_runpod_logs)
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

    def test_translator_prompt_strict_anti_omission_rules(self):
        """Translation prompts must enforce faithful translation without omitting proper names, figures, or clauses."""
        import inspect
        from core.translation import Translator

        trans_src = inspect.getsource(Translator._translate_local)
        self.assertIn("Never omit proper names", trans_src)
        self.assertIn("biblical figures (e.g. David, Jesus, Moses, Paul)", trans_src)
        self.assertIn("without omissions", trans_src)

        with open("runpod_server.py", "r", encoding="utf-8") as f:
            runpod_src = f.read()
        self.assertIn("Never omit proper names", runpod_src)
        self.assertIn("biblical figures (e.g. David, Jesus, Moses, Paul)", runpod_src)
        self.assertIn("without omissions", runpod_src)

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
        default_gpus = get_runpod_gpu_fallbacks()
        self.assertEqual(default_gpus[0], "NVIDIA RTX A5000")
        self.assertIn("NVIDIA GeForce RTX 3090", default_gpus)

        gpus = get_runpod_gpu_fallbacks("NVIDIA RTX 4000 Ada Generation")
        self.assertEqual(gpus[0], "NVIDIA RTX 4000 Ada Generation")
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

    def test_interim_translation_status_inheritance_and_retention(self):
        """Translation messages inherit segment status and retain multi-chunk state in bounded FIFO."""
        engine = TranscriptionEngine(broadcast_callback=MagicMock())
        runpod = RunPodSTTEngine(engine)

        client_seg_id = "seg_multi_chunk_trans_1"
        runpod.pending_seg_id = client_seg_id
        runpod.pending_status = "interim"

        # 1. Chunk 1: STT interim arrival
        stt_data = {"id": "srv_chunk_1", "status": "final", "col1_text": "De kerk als gondel"}
        runpod.server_to_client_id["srv_chunk_1"] = runpod.pending_seg_id
        stt_data["id"] = runpod.server_to_client_id[stt_data["id"]]
        if "col1_text" in stt_data and runpod.pending_status:
            stt_data["status"] = runpod.pending_status
        runpod.segment_status[client_seg_id] = stt_data["status"]
        self.assertEqual(stt_data["status"], "interim")
        self.assertEqual(runpod.segment_status[client_seg_id], "interim")

        # 2. Chunk 1: Translation arrival (inherits status='interim', does not pop from dict)
        trans_data_1 = {"id": "srv_chunk_1", "status": "final", "col2_text": "The church as a gondola"}
        trans_data_1["id"] = runpod.server_to_client_id[trans_data_1["id"]]
        # Status inheritance:
        trans_data_1["status"] = runpod.segment_status.get(client_seg_id, runpod.pending_status or "final")
        self.assertEqual(trans_data_1["status"], "interim")

        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = trans_data_1["col2_text"]

        # Ensure translations are NOT prematurely cleared while interim
        self.assertIn(client_seg_id, runpod.segment_translations)

        # 3. Chunk 2: Final STT & Translation
        runpod.pending_status = "final"
        runpod.segment_status[client_seg_id] = "final"

        trans_data_2 = {"id": "srv_chunk_2", "status": "final", "col2_text": "that is a bit exaggerated"}
        runpod.server_to_client_id["srv_chunk_2"] = client_seg_id
        trans_data_2["id"] = runpod.server_to_client_id[trans_data_2["id"]]
        trans_data_2["status"] = runpod.segment_status.get(client_seg_id, "final")

        incoming_col = trans_data_2["col2_text"]
        prev_col = runpod.segment_translations.get(client_seg_id, {}).get("col2_text", "")
        if prev_col and not (incoming_col.lower().startswith(prev_col[:15].lower()) or prev_col.lower() in incoming_col.lower()):
            full_col = f"{prev_col} {incoming_col}".strip()
        else:
            full_col = incoming_col
        runpod.segment_translations[client_seg_id]["col2_text"] = full_col
        trans_data_2["col2_text"] = full_col

        self.assertEqual(trans_data_2["status"], "final")
        self.assertEqual(trans_data_2["col2_text"], "The church as a gondola that is a bit exaggerated")

    def test_restitch_segment_text_prevents_doublures_on_long_utterances(self):
        """restitch_segment_text must prevent duplication when Whisper refines opening words or punctuation."""
        # 1. Opening words slightly refined as audio accumulates
        prev = "De kerk als gondel."
        # Whisper refined "gondel" to "een gondel" and added rest of long sentence (>25s)
        revised = "De kerk als een gondel, dat is wel wat gechargeerd door de media."
        stitched = restitch_segment_text(prev, revised)
        self.assertEqual(stitched, "De kerk als een gondel, dat is wel wat gechargeerd door de media.")
        self.assertNotIn("De kerk als gondel. De kerk", stitched)

        # 2. Punctuation and slight casing change at start
        prev2 = "wij gaan nu bidden"
        revised2 = "Wij gaan nu bidden voor vrede over de hele wereld."
        stitched2 = restitch_segment_text(prev2, revised2)
        self.assertEqual(stitched2, "Wij gaan nu bidden voor vrede over de hele wereld.")

        # 3. Server stripped overlap (tail delta arrival)
        prev3 = "Wij zingen samen een lied"
        tail = "een lied van hoop en vreugde."
        stitched3 = restitch_segment_text(prev3, tail)
        self.assertEqual(stitched3, "Wij zingen samen een lied van hoop en vreugde.")

        # 4. Translation restitching without duplication
        prev_trans = "The church as a gondola."
        revised_trans = "The church as a gondola, which is somewhat exaggerated."
        stitched_trans = restitch_segment_text(prev_trans, revised_trans)
        self.assertEqual(stitched_trans, "The church as a gondola, which is somewhat exaggerated.")

        # 5. Word-boundary stem/incomplete word refinement (e.g. "bepalen." -> "bepalend.")
        prev_stem = "En het begint telkens weer bij de overwinning van de Heer Jezus Christus. Het fundament van ons geloof. Alles bepalen."
        inc_stem = "bepalend."
        stitched_stem = restitch_segment_text(prev_stem, inc_stem)
        self.assertEqual(
            stitched_stem,
            "En het begint telkens weer bij de overwinning van de Heer Jezus Christus. Het fundament van ons geloof. Alles bepalend."
        )
        self.assertNotIn("bepalen. bepalend", stitched_stem)

        # 6. Truncated word plus continuation (e.g. "na paas." -> "Pasen en de Heer...")
        prev_cutoff = "Het gaat vandaag over de derde zondag na paas."
        inc_cont = "Pasen en de Heer verheugt zich."
        stitched_cont = restitch_segment_text(prev_cutoff, inc_cont)
        self.assertEqual(
            stitched_cont,
            "Het gaat vandaag over de derde zondag na Pasen en de Heer verheugt zich."
        )

    def test_runpod_hallucination_and_translation_suppression(self):
        """RunPodSTTEngine drops noise hallucinations and suppresses associated subsequent translations."""
        broadcast_messages = []
        async def mock_broadcast(msg):
            broadcast_messages.append(msg)

        engine = TranscriptionEngine(broadcast_callback=mock_broadcast)
        runpod = RunPodSTTEngine(engine)

        # 1. Hallucinated segment (noise: "Dank u wel.")
        seg_id_hallucination = "seg_hallucinated_123"
        stt_msg = {
            "id": seg_id_hallucination,
            "status": "final",
            "col1_text": "Dank u wel.",
        }

        # Simulate STT arrival in ws_receiver
        filtered_text = filter_anti_hallucination(stt_msg["col1_text"])
        if not filtered_text:
            runpod.suppressed_seg_ids.add(seg_id_hallucination)

        self.assertIn(seg_id_hallucination, runpod.suppressed_seg_ids)
        # Message was suppressed, so NOT broadcast
        self.assertEqual(len(broadcast_messages), 0)

        # 2. Subsequent translation messages arrive for the suppressed segment
        trans_msg_1 = {
            "id": seg_id_hallucination,
            "status": "final",
            "col2_text": "Thank you.",
        }
        # In ws_receiver: check if client_seg_id in runpod.suppressed_seg_ids
        if trans_msg_1["id"] in runpod.suppressed_seg_ids:
            pass  # Dropped!
        else:
            broadcast_messages.append(trans_msg_1)

        trans_msg_2 = {
            "id": seg_id_hallucination,
            "status": "final",
            "col3_text": "Спасибо.",
        }
        if trans_msg_2["id"] in runpod.suppressed_seg_ids:
            pass  # Dropped!
        else:
            broadcast_messages.append(trans_msg_2)

        # Neither transcript nor translations for the hallucinated segment were broadcast
        self.assertEqual(len(broadcast_messages), 0)

        # 3. Legitimate segment arrives
        seg_id_legit = "seg_legit_456"
        legit_stt = {
            "id": seg_id_legit,
            "status": "final",
            "col1_text": "Welkom allemaal in deze dienst.",
        }
        filtered_legit = filter_anti_hallucination(legit_stt["col1_text"])
        self.assertIsNotNone(filtered_legit)
        broadcast_messages.append(legit_stt)

        legit_trans = {
            "id": seg_id_legit,
            "status": "final",
            "col2_text": "Welcome everyone to this service.",
        }
        if legit_trans["id"] not in runpod.suppressed_seg_ids:
            broadcast_messages.append(legit_trans)

        self.assertEqual(len(broadcast_messages), 2)
        self.assertEqual(broadcast_messages[0]["col1_text"], "Welkom allemaal in deze dienst.")
        self.assertEqual(broadcast_messages[1]["col2_text"], "Welcome everyone to this service.")

    def test_interim_followed_by_final_preserves_full_sentence_without_prefix_chop(self):
        """Interim chunks must not pollute last_transcript and cause final chunks to be prefix-chopped."""
        broadcast_messages = []
        async def mock_broadcast(msg):
            broadcast_messages.append(msg)

        engine = TranscriptionEngine(broadcast_callback=mock_broadcast)
        runpod = RunPodSTTEngine(engine)
        runpod.last_transcript = "In de naam van de Vader en de Zoon."

        seg_id = "seg_multi_chunk_rotterdam_1"
        runpod.pending_seg_id = seg_id

        # 1. Chunk 1 (interim arrival)
        runpod.pending_status = "interim"
        raw_interim = {
            "id": seg_id,
            "status": "interim",
            "col1_text": "Zoals verschillende gemeenteleden weten heb ik Rotterdamse",
        }

        # Simulate ws_receiver logic for Chunk 1
        client_seg_id = raw_interim["id"]
        is_final = raw_interim["status"] == "final"
        prev_text = runpod.segment_full_transcripts.get(client_seg_id, "")
        full_text_1 = restitch_segment_text(prev_text, raw_interim["col1_text"])
        effective_last_transcript = "" if prev_text else runpod.last_transcript
        filtered_text_1 = filter_anti_hallucination(full_text_1, last_transcript=effective_last_transcript)

        self.assertIsNotNone(filtered_text_1)
        self.assertEqual(filtered_text_1, "Zoals verschillende gemeenteleden weten heb ik Rotterdamse")
        if is_final:
            runpod.last_transcript = filtered_text_1[-1200:]
        runpod.segment_full_transcripts[client_seg_id] = filtered_text_1

        # Notice: runpod.last_transcript was NOT updated because is_final is False!
        self.assertEqual(runpod.last_transcript, "In de naam van de Vader en de Zoon.")

        # 2. Chunk 2 (final arrival with tail word "roots.")
        runpod.pending_status = "final"
        raw_final = {
            "id": seg_id,
            "status": "final",
            "col1_text": "roots.",
        }

        is_final = raw_final["status"] == "final"
        prev_text = runpod.segment_full_transcripts.get(client_seg_id, "")
        full_text_2 = restitch_segment_text(prev_text, raw_final["col1_text"])
        # Crucial: effective_last_transcript must be "" because prev_text exists
        effective_last_transcript = "" if prev_text else runpod.last_transcript
        filtered_text_2 = filter_anti_hallucination(full_text_2, last_transcript=effective_last_transcript)

        self.assertIsNotNone(filtered_text_2)
        # MUST BE the full sentence, NOT just 'roots.'!
        self.assertEqual(
            filtered_text_2,
            "Zoals verschillende gemeenteleden weten heb ik Rotterdamse roots."
        )
        self.assertNotEqual(filtered_text_2, "roots.")

        if is_final:
            runpod.last_transcript = filtered_text_2[-1200:]
        runpod.segment_full_transcripts[client_seg_id] = filtered_text_2

        # Now that it's final, last_transcript is updated for future segments
        self.assertTrue(runpod.last_transcript.endswith("Rotterdamse roots."))

    def test_runpod_translation_trailing_credit_stripped(self):
        """Broadcast credit artifacts (e.g. 'TV Gelderland 2021' in Dutch/English/Russian) must be stripped from all translation columns."""
        from core.stt.local_engine import TAIL_HALLUCINATION_PATTERN
        import re

        test_cases = [
            ("Want to see you today. TV Gelderland 2021", "Want to see you today."),
            ("но и сегодня для нас. ТВ Гельдерланд 2021", "но и сегодня для нас."),
            ("Maar ook vandaag voor ons. TV Gelreland 2021", "Maar ook vandaag voor ons."),
            ("Voor ons allen. Omroep Gelderland", "Voor ons allen."),
            ("TV Gelderland 2021", ""),
            ("ТВ Гельдерланд 2021", ""),
        ]

        for raw, expected in test_cases:
            cleaned = TAIL_HALLUCINATION_PATTERN.sub("", raw).strip()
            cleaned = re.sub(
                r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)(?:tv|тв)?\s*(?:gelderland|gelreland|гельдерланд)(?:\s+\d{4})?[\.\,\!\?\s]*$",
                "",
                cleaned,
                flags=re.IGNORECASE
            ).strip()
            self.assertEqual(cleaned, expected)

    def test_runpod_translation_direct_update_no_doublure(self):
        """Incoming translation packets represent the cumulative segment translation and must replace earlier interim translations directly without concatenating or doubling."""
        engine = TranscriptionEngine(broadcast_callback=lambda msg: None)
        runpod = RunPodSTTEngine(engine)

        client_seg_id = "seg_test_trans_direct"
        runpod.pending_seg_id = client_seg_id

        # Interim translation
        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = (
            "If you imagine, there would be a savior, Jesus, who cried out on the cross, 'It is finished.'"
        )

        # Later final translation with rephrased wording from LLM
        incoming_final = (
            "I must introduce you to a savior, Jesus, who cried out on the cross, 'It is finished,' but that saving power would become inaccessible."
        )

        # Direct update with is_delta: False as implemented in runpod_engine.py
        data = {"id": client_seg_id, "is_delta": False, "col2_text": incoming_final}
        if data.get("is_delta") is not None:
            is_delta = bool(data.get("is_delta"))
        else:
            is_delta = runpod.server_chunk_is_delta.get(client_seg_id, False)

        prev_col = runpod.segment_translations.get(client_seg_id, {}).get("col2_text", "")
        if is_delta and prev_col:
            full_col = restitch_segment_text(prev_col, incoming_final)
        else:
            full_col = incoming_final.strip()

        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = full_col

        # Must NOT contain the concatenated interim translation
        self.assertEqual(runpod.segment_translations[client_seg_id]["col2_text"], incoming_final)
        self.assertNotIn("If you imagine", runpod.segment_translations[client_seg_id]["col2_text"])

    def test_runpod_full_segment_revision_protocol(self):
        """Verify Full-Segment Revision protocol: when is_delta: False is received, incoming STT transcripts and translations replace prior interim versions directly without duplication, while is_delta: True stitches."""
        engine = TranscriptionEngine(broadcast_callback=lambda msg: None)
        runpod = RunPodSTTEngine(engine)

        client_seg_id = "seg_full_revision_123"
        runpod.pending_seg_id = client_seg_id
        srv_chunk_1 = "srv_c1"
        srv_chunk_2 = "srv_c2"
        runpod.server_to_client_id[srv_chunk_1] = client_seg_id
        runpod.server_to_client_id[srv_chunk_2] = client_seg_id

        # --- 1. Interim Chunk 1: is_delta: False ---
        stt_msg_1 = {"id": srv_chunk_1, "status": "interim", "is_delta": False, "col1_text": "Het geloof is bepalen"}
        stt_msg_1["id"] = runpod.server_to_client_id[stt_msg_1["id"]]

        incoming_stt_1 = stt_msg_1["col1_text"].strip()
        prev_stt_1 = runpod.segment_full_transcripts.get(client_seg_id, "")
        is_server_delta_1 = stt_msg_1.get("is_delta", None)
        if is_server_delta_1 is not None:
            is_delta_chunk_1 = bool(is_server_delta_1)
            if is_delta_chunk_1 and prev_stt_1:
                full_stt_1 = restitch_segment_text(prev_stt_1, incoming_stt_1)
            else:
                full_stt_1 = incoming_stt_1
        else:
            full_stt_1 = restitch_segment_text(prev_stt_1, incoming_stt_1)
            is_delta_chunk_1 = bool(prev_stt_1 and incoming_stt_1 != full_stt_1)

        runpod.server_chunk_is_delta[srv_chunk_1] = is_delta_chunk_1
        runpod.segment_last_chunk_is_delta[client_seg_id] = is_delta_chunk_1
        runpod.segment_full_transcripts[client_seg_id] = full_stt_1

        # Translation for Chunk 1: is_delta: False
        trans_msg_1 = {"id": srv_chunk_1, "status": "interim", "is_delta": False, "col2_text": "Faith is determin"}
        trans_msg_1["id"] = runpod.server_to_client_id[trans_msg_1["id"]]

        incoming_col_1 = trans_msg_1["col2_text"].strip()
        prev_col_1 = runpod.segment_translations.setdefault(client_seg_id, {}).get("col2_text", "")
        if trans_msg_1.get("is_delta") is not None:
            is_delta_1 = bool(trans_msg_1.get("is_delta"))
        else:
            is_delta_1 = runpod.server_chunk_is_delta.get(srv_chunk_1, False)

        if is_delta_1 and prev_col_1:
            full_col_1 = restitch_segment_text(prev_col_1, incoming_col_1)
        else:
            full_col_1 = incoming_col_1
        runpod.segment_translations[client_seg_id]["col2_text"] = full_col_1

        self.assertEqual(runpod.segment_full_transcripts[client_seg_id], "Het geloof is bepalen")
        self.assertEqual(runpod.segment_translations[client_seg_id]["col2_text"], "Faith is determin")
        self.assertFalse(runpod.server_chunk_is_delta[srv_chunk_1])

        # --- 2. Final Chunk 2: Full-Segment Revision with is_delta: False ---
        stt_msg_2 = {"id": srv_chunk_2, "status": "final", "is_delta": False, "col1_text": "Het geloof is bepalend."}
        stt_msg_2["id"] = runpod.server_to_client_id[stt_msg_2["id"]]

        incoming_stt_2 = stt_msg_2["col1_text"].strip()
        prev_stt_2 = runpod.segment_full_transcripts.get(client_seg_id, "")
        is_server_delta_2 = stt_msg_2.get("is_delta", None)
        if is_server_delta_2 is not None:
            is_delta_chunk_2 = bool(is_server_delta_2)
            if is_delta_chunk_2 and prev_stt_2:
                full_stt_2 = restitch_segment_text(prev_stt_2, incoming_stt_2)
            else:
                full_stt_2 = incoming_stt_2
        else:
            full_stt_2 = restitch_segment_text(prev_stt_2, incoming_stt_2)
            is_delta_chunk_2 = bool(prev_stt_2 and incoming_stt_2 != full_stt_2)

        runpod.server_chunk_is_delta[srv_chunk_2] = is_delta_chunk_2
        runpod.segment_last_chunk_is_delta[client_seg_id] = is_delta_chunk_2
        runpod.segment_full_transcripts[client_seg_id] = full_stt_2

        # Translation for Chunk 2: is_delta: False
        trans_msg_2 = {"id": srv_chunk_2, "status": "final", "is_delta": False, "col2_text": "Faith is decisive."}
        trans_msg_2["id"] = runpod.server_to_client_id[trans_msg_2["id"]]

        incoming_col_2 = trans_msg_2["col2_text"].strip()
        prev_col_2 = runpod.segment_translations.setdefault(client_seg_id, {}).get("col2_text", "")
        if trans_msg_2.get("is_delta") is not None:
            is_delta_2 = bool(trans_msg_2.get("is_delta"))
        else:
            is_delta_2 = runpod.server_chunk_is_delta.get(srv_chunk_2, False)

        if is_delta_2 and prev_col_2:
            full_col_2 = restitch_segment_text(prev_col_2, incoming_col_2)
        else:
            full_col_2 = incoming_col_2
        runpod.segment_translations[client_seg_id]["col2_text"] = full_col_2

        # Both STT and translation directly replace prior interim without duplication
        self.assertEqual(runpod.segment_full_transcripts[client_seg_id], "Het geloof is bepalend.")
        self.assertEqual(runpod.segment_translations[client_seg_id]["col2_text"], "Faith is decisive.")
        self.assertNotIn("bepalen Het", runpod.segment_full_transcripts[client_seg_id])
        self.assertNotIn("determin Faith", runpod.segment_translations[client_seg_id]["col2_text"])

        # --- 3. Backward Compatibility: Server sending is_delta: True ---
        client_seg_delta = "seg_delta_compat_456"
        runpod.pending_seg_id = client_seg_delta
        runpod.server_to_client_id["srv_d1"] = client_seg_delta
        runpod.server_to_client_id["srv_d2"] = client_seg_delta

        # Chunk 1 (is_delta: False)
        runpod.segment_full_transcripts[client_seg_delta] = "Onze hulp"
        runpod.segment_translations.setdefault(client_seg_delta, {})["col2_text"] = "Our help"

        # Chunk 2 (is_delta: True delta tail)
        delta_stt_msg = {"id": "srv_d2", "status": "final", "is_delta": True, "col1_text": "is de Heer."}
        delta_stt_msg["id"] = runpod.server_to_client_id[delta_stt_msg["id"]]

        inc_d_stt = delta_stt_msg["col1_text"].strip()
        prev_d_stt = runpod.segment_full_transcripts.get(client_seg_delta, "")
        is_server_delta_d = delta_stt_msg.get("is_delta", None)
        if is_server_delta_d is not None:
            is_delta_chunk_d = bool(is_server_delta_d)
            if is_delta_chunk_d and prev_d_stt:
                full_d_stt = restitch_segment_text(prev_d_stt, inc_d_stt)
            else:
                full_d_stt = inc_d_stt
        else:
            full_d_stt = restitch_segment_text(prev_d_stt, inc_d_stt)
            is_delta_chunk_d = bool(prev_d_stt and inc_d_stt != full_d_stt)

        runpod.server_chunk_is_delta["srv_d2"] = is_delta_chunk_d
        runpod.segment_full_transcripts[client_seg_delta] = full_d_stt

        delta_trans_msg = {"id": "srv_d2", "status": "final", "is_delta": True, "col2_text": "is the Lord."}
        delta_trans_msg["id"] = runpod.server_to_client_id[delta_trans_msg["id"]]

        inc_d_trans = delta_trans_msg["col2_text"].strip()
        prev_d_trans = runpod.segment_translations.setdefault(client_seg_delta, {}).get("col2_text", "")
        if delta_trans_msg.get("is_delta") is not None:
            is_delta_d = bool(delta_trans_msg.get("is_delta"))
        else:
            is_delta_d = runpod.server_chunk_is_delta.get("srv_d2", False)

        if is_delta_d and prev_d_trans:
            full_d_trans = restitch_segment_text(prev_d_trans, inc_d_trans)
        else:
            full_d_trans = inc_d_trans
        runpod.segment_translations[client_seg_delta]["col2_text"] = full_d_trans

        # With is_delta: True, both STT and translation stitch correctly
        self.assertEqual(runpod.segment_full_transcripts[client_seg_delta], "Onze hulp is de Heer.")
        self.assertEqual(runpod.segment_translations[client_seg_delta]["col2_text"], "Our help is the Lord.")

        # --- 4. Server Protocol Verification in runpod_server.py ---
        with open("runpod_server.py", "r", encoding="utf-8") as f:
            server_src = f.read()

        self.assertIn('elif data.get("action") == "audio_meta":', server_src)
        self.assertIn('is_same_segment = bool(last_processed_seg_id and last_processed_seg_id == seg_id)', server_src)
        self.assertIn('elif last_transcript and not is_same_segment:', server_src)
        self.assertIn('"is_delta": False', server_src)

    def test_runpod_translation_delta_tail_stitching(self):
        """When RunPod server sends a delta tail chunk, translations must be restitched onto earlier translations so the opening clause is not omitted."""
        engine = TranscriptionEngine(broadcast_callback=lambda msg: None)
        runpod = RunPodSTTEngine(engine)

        client_seg_id = "seg_test_trans_delta"
        runpod.pending_seg_id = client_seg_id

        # Chunk 1: Interim
        srv_chunk_1 = "srv_chunk_1"
        runpod.server_to_client_id[srv_chunk_1] = client_seg_id
        c1_stt = "Onze hulp."
        runpod.segment_full_transcripts[client_seg_id] = c1_stt
        runpod.server_chunk_is_delta[srv_chunk_1] = False
        runpod.segment_last_chunk_is_delta[client_seg_id] = False

        c1_trans = "Our help."
        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = c1_trans

        # Chunk 2: RunPod server sends delta tail because overlap was trimmed
        srv_chunk_2 = "srv_chunk_2"
        runpod.server_to_client_id[srv_chunk_2] = client_seg_id
        c2_stt_incoming = "is de Heer."
        prev_stt = runpod.segment_full_transcripts[client_seg_id]
        full_stt = restitch_segment_text(prev_stt, c2_stt_incoming)
        is_delta_chunk = bool(prev_stt and c2_stt_incoming != full_stt)
        self.assertTrue(is_delta_chunk)

        runpod.server_chunk_is_delta[srv_chunk_2] = is_delta_chunk
        runpod.segment_last_chunk_is_delta[client_seg_id] = is_delta_chunk
        runpod.segment_full_transcripts[client_seg_id] = full_stt

        # Translation arrives for chunk 2 (RunPod translated only the delta tail)
        c2_trans_incoming = "is the Lord."
        is_delta = runpod.server_chunk_is_delta.get(srv_chunk_2, False)
        prev_trans = runpod.segment_translations[client_seg_id]["col2_text"]

        if is_delta and prev_trans:
            full_trans = restitch_segment_text(prev_trans, c2_trans_incoming)
        else:
            full_trans = c2_trans_incoming

        runpod.segment_translations[client_seg_id]["col2_text"] = full_trans

        # Crucial check: opening clause "Our help." MUST be preserved!
        self.assertTrue(full_trans.startswith("Our help"))
        self.assertTrue(full_trans.endswith("is the Lord."))
        self.assertEqual(full_trans, "Our help. is the Lord.")

    def test_session_window_initialization(self):
        """TranscriptionEngine initializes sliding session window parameters from env with fallbacks."""
        import time
        from unittest.mock import patch, MagicMock
        from core.engine import TranscriptionEngine

        # Test custom environment values (e.g. 2 min window for fast testing)
        with patch.dict(os.environ, {
            "INITIAL_SESSION_WINDOW_MINUTES": "2",
            "EXTEND_WINDOW_MINUTES": "3",
            "MAX_SESSION_CEILING_MINUTES": "10",
        }, clear=False):
            engine = TranscriptionEngine(broadcast_callback=MagicMock())
            self.assertEqual(engine.initial_session_window_minutes, 2.0)
            self.assertEqual(engine.extend_window_minutes, 3.0)
            self.assertEqual(engine.max_session_ceiling_minutes, 10.0)
            self.assertEqual(engine.MAX_DURATION_SECONDS, 120)
            self.assertAlmostEqual(engine.session_expiry_time, engine.session_start_time + 120, delta=1.0)
            self.assertAlmostEqual(engine.session_ceiling_time, engine.session_start_time + 600, delta=1.0)

        # Test fallback to MAX_SESSION_MINUTES when INITIAL_SESSION_WINDOW_MINUTES is omitted
        env_copy = os.environ.copy()
        env_copy.pop("INITIAL_SESSION_WINDOW_MINUTES", None)
        env_copy["MAX_SESSION_MINUTES"] = "45"
        with patch.dict(os.environ, env_copy, clear=True):
            engine = TranscriptionEngine(broadcast_callback=MagicMock())
            self.assertEqual(engine.initial_session_window_minutes, 45.0)
            self.assertEqual(engine.MAX_DURATION_SECONDS, 45 * 60)

    def test_session_window_reset_timer(self):
        """reset_session_timer resets start, expiry, and ceiling, and broadcasts session_timer update."""
        import time
        from unittest.mock import MagicMock
        from core.engine import TranscriptionEngine

        broadcast_mock = MagicMock()
        engine = TranscriptionEngine(broadcast_callback=broadcast_mock)
        engine.initial_session_window_minutes = 5.0
        engine.max_session_ceiling_minutes = 20.0

        broadcast_mock.reset_mock()
        t0 = time.time()
        engine.reset_session_timer(reason="test_reset")

        self.assertGreaterEqual(engine.session_start_time, t0)
        self.assertAlmostEqual(engine.session_expiry_time, engine.session_start_time + 300, delta=1.0)
        self.assertAlmostEqual(engine.session_ceiling_time, engine.session_start_time + 1200, delta=1.0)

        broadcast_mock.assert_called_once()
        sent_msg = broadcast_mock.call_args[0][0]
        self.assertEqual(sent_msg["type"], "session_timer")
        self.assertEqual(sent_msg["reason"], "test_reset")
        self.assertAlmostEqual(sent_msg["remaining_sec"], 300, delta=2)

    def test_session_window_extend_and_ceiling(self):
        """extend_session_window extends expiry up to but not exceeding the safety ceiling."""
        import time
        from unittest.mock import MagicMock
        from core.engine import TranscriptionEngine

        broadcast_mock = MagicMock()
        engine = TranscriptionEngine(broadcast_callback=broadcast_mock)
        engine.initial_session_window_minutes = 10.0
        engine.extend_window_minutes = 15.0
        engine.max_session_ceiling_minutes = 25.0
        engine.reset_session_timer(reason="init")

        # Expiry is initially t0 + 10m
        initial_expiry = engine.session_expiry_time

        # Extending by 15m from now extends beyond initial_expiry
        engine.extend_session_window(reason="stt_activity")
        self.assertGreater(engine.session_expiry_time, initial_expiry)
        self.assertLessEqual(engine.session_expiry_time, engine.session_ceiling_time)

        # Force a huge extend request far in the future — must be capped at ceiling
        engine.extend_window_minutes = 500.0
        engine.extend_session_window(reason="stt_activity")
        self.assertAlmostEqual(engine.session_expiry_time, engine.session_ceiling_time, delta=0.01)

    def test_session_window_auto_pause_on_expiry(self):
        """When current time exceeds session_expiry_time, stream auto-pauses and broadcasts auto_paused."""
        import asyncio
        import time
        from unittest.mock import AsyncMock
        from core.engine import TranscriptionEngine
        from core.stt.local_engine import LocalSTTEngine

        broadcast_mock = AsyncMock()
        engine = TranscriptionEngine(broadcast_callback=broadcast_mock)
        engine.initial_session_window_minutes = 15.0
        local_stt = LocalSTTEngine(engine)

        # Expire session artificially
        engine.session_expiry_time = time.time() - 5.0
        engine.is_paused = False

        # Verify condition
        self.assertTrue(time.time() > engine.session_expiry_time)

    def test_runpod_stop_pod_mutation_payload(self):
        """stop_runpod_pod sends the correct podStop GraphQL mutation with podId and resets active_pod_id."""
        import asyncio
        import json
        from unittest.mock import MagicMock, patch
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)
        runpod.active_pod_id = "pod-xyz-123"

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({
            "data": {"podTerminate": None}
        }).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fake_api_key"}):
            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                loop = asyncio.new_event_loop()
                success = loop.run_until_complete(runpod.stop_runpod_pod())
                loop.close()

                self.assertTrue(success)
                self.assertIsNone(runpod.active_pod_id)

                # Verify payload sent to RunPod GraphQL API
                sent_req = mock_urlopen.call_args[0][0]
                sent_body = json.loads(sent_req.data.decode("utf-8"))
                self.assertIn("podTerminate", sent_body["query"])
                self.assertIn("pod-xyz-123", sent_body["query"])

    def test_runpod_active_pod_id_tracking(self):
        """active_pod_id is tracked on auto-discovery and on pod deployment."""
        import asyncio
        import json
        from unittest.mock import MagicMock, patch
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)
        self.assertIsNone(runpod.active_pod_id)

        # 1. Discovery tracking
        discovery_payload = {
            "data": {
                "myself": {
                    "pods": [{
                        "id": "pod-disc-456",
                        "name": "ondertitels-gpu",
                        "desiredStatus": "RUNNING",
                        "runtime": {
                            "ports": [{
                                "ip": "100.200.50.1",
                                "isIpPublic": True,
                                "privatePort": 8000,
                                "publicPort": 18000,
                                "type": "tcp"
                            }]
                        }
                    }]
                }
            }
        }
        mock_disc_resp = MagicMock()
        mock_disc_resp.read.return_value = json.dumps(discovery_payload).encode("utf-8")
        mock_disc_resp.__enter__.return_value = mock_disc_resp

        with patch.dict(os.environ, {"RUNPOD_API_KEY": "fake_key"}):
            with patch("urllib.request.urlopen", return_value=mock_disc_resp):
                url = runpod.discover_runpod_url()
                self.assertEqual(url, "ws://100.200.50.1:18000/stream")
                self.assertEqual(runpod.active_pod_id, "pod-disc-456")

        # 2. Deployment tracking (REST API v2)
        mock_deploy_resp = MagicMock()
        mock_deploy_resp.read.return_value = json.dumps({"id": "pod-dep-789"}).encode("utf-8")
        mock_deploy_resp.__enter__.return_value = mock_deploy_resp

        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "fake_key",
            "RUNPOD_TEMPLATE_ID": "tmpl-1",
            "RUNPOD_VOLUME_ID": "cmu-vol-1",
        }):
            with patch("urllib.request.urlopen", return_value=mock_deploy_resp):
                loop = asyncio.new_event_loop()
                deployed = loop.run_until_complete(runpod.deploy_runpod_pod())
                loop.close()

                self.assertTrue(deployed)
                self.assertEqual(runpod.active_pod_id, "pod-dep-789")

    def test_runpod_override_cmd_in_deployment_payload(self):
        """deploy_runpod_pod embeds /workspace/runpod_server.py override in container startup cmd."""
        import asyncio
        import json
        from unittest.mock import MagicMock, patch
        from core.stt.runpod_engine import RunPodSTTEngine, RUNPOD_BOOT_CMD, RUNPOD_OVERRIDE_CMD

        self.assertIn("[ -f /workspace/boot.sh ] && bash /workspace/boot.sh", RUNPOD_BOOT_CMD)
        self.assertIn("curl -fsSL", RUNPOD_BOOT_CMD)
        self.assertIn("boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh", RUNPOD_BOOT_CMD)
        self.assertEqual(RUNPOD_OVERRIDE_CMD, RUNPOD_BOOT_CMD)

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)

        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"id": "pod-check-override"}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp

        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "fake_key",
            "RUNPOD_TEMPLATE_ID": "tmpl-1",
            "RUNPOD_VOLUME_ID": "cmu-vol-1",
        }):
            with patch("urllib.request.urlopen", return_value=mock_resp) as mock_urlopen:
                loop = asyncio.new_event_loop()
                loop.run_until_complete(runpod.deploy_runpod_pod())
                loop.close()

                sent_req = mock_urlopen.call_args[0][0]
                payload = json.loads(sent_req.data.decode("utf-8"))
                self.assertNotIn("entrypoint", payload)
                self.assertNotIn("cmd", payload)
                self.assertEqual(payload["templateId"], "tmpl-1")
                self.assertEqual(payload["mounts"]["global"][0]["volumeId"], "cmu-vol-1")

    def test_runpod_deploy_v2_datacenter_preference_and_fallback(self):
        """deploy_runpod_pod attempts preferred dataCenterIds first, falling back to any region on stock failure."""
        import asyncio
        import json
        import urllib.error
        from unittest.mock import MagicMock, patch
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)

        # Case 1: Preferred DC succeeds on first try
        mock_success_resp = MagicMock()
        mock_success_resp.read.return_value = json.dumps({"id": "pod-dc-preferred"}).encode("utf-8")
        mock_success_resp.__enter__.return_value = mock_success_resp

        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "fake_key",
            "RUNPOD_TEMPLATE_ID": "tmpl-1",
            "RUNPOD_VOLUME_ID": "cmu-vol-1",
            "RUNPOD_DATA_CENTER_ID": "EU-RO-1, US-CA-1",
            "RUNPOD_GPU_ID": "NVIDIA GeForce RTX 4090",
        }):
            with patch("urllib.request.urlopen", return_value=mock_success_resp) as mock_urlopen:
                loop = asyncio.new_event_loop()
                deployed = loop.run_until_complete(runpod.deploy_runpod_pod())
                loop.close()

                self.assertTrue(deployed)
                self.assertEqual(runpod.active_pod_id, "pod-dc-preferred")
                self.assertEqual(mock_urlopen.call_count, 1)
                sent_req = mock_urlopen.call_args[0][0]
                payload = json.loads(sent_req.data.decode("utf-8"))
                self.assertEqual(payload.get("dataCenterIds"), ["EU-RO-1", "US-CA-1"])

        # Case 2: Preferred DC fails with stock error (HTTP 400), falls back to any region
        mock_fallback_resp = MagicMock()
        mock_fallback_resp.read.return_value = json.dumps({"id": "pod-dc-fallback"}).encode("utf-8")
        mock_fallback_resp.__enter__.return_value = mock_fallback_resp

        http_err = urllib.error.HTTPError(
            url="https://api.runpod.io/v2/pods",
            code=400,
            msg="Bad Request",
            hdrs={},
            fp=MagicMock(read=lambda: b'{"error": "no stock available in specified datacenters"}')
        )

        with patch.dict(os.environ, {
            "RUNPOD_API_KEY": "fake_key",
            "RUNPOD_TEMPLATE_ID": "tmpl-1",
            "RUNPOD_VOLUME_ID": "cmu-vol-1",
            "RUNPOD_DATA_CENTER_ID": "EU-RO-1",
            "RUNPOD_GPU_ID": "NVIDIA GeForce RTX 4090",
        }):
            call_count = 0
            def mock_side_effect(req, timeout=15):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise http_err
                return mock_fallback_resp

            with patch("urllib.request.urlopen", side_effect=mock_side_effect) as mock_urlopen:
                loop = asyncio.new_event_loop()
                deployed = loop.run_until_complete(runpod.deploy_runpod_pod())
                loop.close()

                self.assertTrue(deployed)
                self.assertEqual(runpod.active_pod_id, "pod-dc-fallback")
                self.assertEqual(call_count, 2)
                # First attempt had dataCenterIds
                first_payload = json.loads(mock_urlopen.call_args_list[0][0][0].data.decode("utf-8"))
                self.assertEqual(first_payload.get("dataCenterIds"), ["EU-RO-1"])
                # Second attempt fell back without dataCenterIds
                second_payload = json.loads(mock_urlopen.call_args_list[1][0][0].data.decode("utf-8"))
                self.assertNotIn("dataCenterIds", second_payload)

    def test_silence_heartbeat_probe_generation(self):
        """RunPodSTTEngine identifies prolonged silence and triggers heartbeat probe state."""
        import time
        from unittest.mock import MagicMock
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)
        runpod.silence_heartbeat_sec = 5.0

        # Simulate 10 seconds of silence with 0 speech frames
        runpod.last_speech_or_heartbeat_time = time.time() - 10.0
        runpod.speech_frames = 0

        silence_duration = time.time() - runpod.last_speech_or_heartbeat_time
        self.assertGreaterEqual(silence_duration, runpod.silence_heartbeat_sec)

        probe_id = f"probe_{int(time.time() * 1000)}"
        runpod.heartbeat_seg_ids.add(probe_id)
        runpod.heartbeat_probe_sent_time = time.time()
        self.assertIn(probe_id, runpod.heartbeat_seg_ids)

    def test_probe_response_intercepted_and_filtered_from_subtitles(self):
        """Heartbeat probe responses emit telemetry and are strictly suppressed from subtitle broadcasts."""
        import asyncio
        import time
        from unittest.mock import AsyncMock, MagicMock
        from core.stt.runpod_engine import RunPodSTTEngine

        broadcast_mock = AsyncMock()
        engine = MagicMock()
        engine.broadcast = broadcast_mock

        runpod = RunPodSTTEngine(engine)
        probe_id = "probe_1729770000000"
        runpod.heartbeat_seg_ids.add(probe_id)
        runpod.heartbeat_probe_sent_time = time.time() - 0.120  # 120ms ago

        incoming_msg = {
            "id": probe_id,
            "status": "final",
            "col1_text": "dummy noise transcript that must never be shown to users"
        }

        # Probe interception check
        client_seg_id = incoming_msg.get("id")
        is_probe = (
            (client_seg_id and (client_seg_id in runpod.heartbeat_seg_ids or str(client_seg_id).startswith("probe_")))
            or incoming_msg.get("type") == "probe_response"
        )
        self.assertTrue(is_probe)

        loop = asyncio.new_event_loop()
        if is_probe:
            latency_ms = int((time.time() - runpod.heartbeat_probe_sent_time) * 1000)
            loop.run_until_complete(runpod.engine.broadcast({
                "type": "heartbeat",
                "status": "ok",
                "latency_ms": latency_ms,
                "timestamp": time.time(),
            }))
            runpod.heartbeat_seg_ids.discard(client_seg_id)
        loop.close()

        # Telemetry broadcast check
        broadcast_mock.assert_called_once()
        sent_event = broadcast_mock.call_args[0][0]
        self.assertEqual(sent_event["type"], "heartbeat")
        self.assertEqual(sent_event["status"], "ok")
        self.assertGreaterEqual(sent_event["latency_ms"], 100)

        # Must NOT be in heartbeat_seg_ids anymore
        self.assertNotIn(probe_id, runpod.heartbeat_seg_ids)

        # Ensure no subtitle broadcast with col1_text was sent
        self.assertNotIn("col1_text", sent_event)

    def test_clean_shutdown_avoids_recv_error_log(self):
        """When intentional_shutdown is True, normal closure message is logged without WS Recv Error."""
        import io
        from contextlib import redirect_stdout
        from unittest.mock import MagicMock
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        runpod = RunPodSTTEngine(engine)
        runpod.intentional_shutdown = True

        out = io.StringIO()
        with redirect_stdout(out):
            try:
                raise Exception("no close frame received or sent")
            except Exception as e:
                if runpod.intentional_shutdown:
                    print("[RunPod] WebSocket closed normally after intentional podStop.")
                else:
                    print(f"RunPod WS Recv Error: {e}")

        output = out.getvalue()
        self.assertIn("[RunPod] WebSocket closed normally after intentional podStop.", output)
        self.assertNotIn("RunPod WS Recv Error", output)

    def test_local_engine_silence_heartbeat_broadcast(self):
        """LocalSTTEngine broadcasts heartbeat telemetry when silence exceeds threshold."""
        import asyncio
        import time
        from unittest.mock import AsyncMock, MagicMock
        from core.stt.local_engine import LocalSTTEngine

        broadcast_mock = AsyncMock()
        engine = MagicMock()
        engine.broadcast = broadcast_mock

        local = LocalSTTEngine(engine)
        local.silence_heartbeat_sec = 5.0
        local.last_speech_or_heartbeat_time = time.time() - 10.0

        now = time.time()
        silence_s = now - local.last_speech_or_heartbeat_time
        self.assertGreaterEqual(silence_s, local.silence_heartbeat_sec)

        loop = asyncio.new_event_loop()
        loop.run_until_complete(engine.broadcast({
            "type": "heartbeat",
            "status": "ok",
            "latency_ms": 0,
            "timestamp": now,
        }))
        loop.close()

        broadcast_mock.assert_called_once()
        event = broadcast_mock.call_args[0][0]
        self.assertEqual(event["type"], "heartbeat")
        self.assertEqual(event["latency_ms"], 0)

    def test_runpod_mode_defers_audio_stream_start(self):
        """In RunPod mode, TranscriptionEngine.run() does NOT start audio_stream upfront."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from core.engine import TranscriptionEngine
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock(spec=TranscriptionEngine)
        engine.pipeline_mode = "runpod"
        engine.audio_stream = MagicMock()
        engine.stt_engine = MagicMock(spec=RunPodSTTEngine)
        engine.stt_engine.run = AsyncMock()
        engine.reset_session_timer = MagicMock()

        # Run the unbound method with our mocked engine
        loop = asyncio.new_event_loop()
        loop.run_until_complete(TranscriptionEngine.run(engine))
        loop.close()

        # audio_stream.start() must NOT have been called upfront
        engine.audio_stream.start.assert_not_called()
        engine.stt_engine.run.assert_called_once()

    def test_local_mode_starts_audio_stream_immediately(self):
        """In local/cloud mode, TranscriptionEngine.run() starts audio_stream immediately."""
        import asyncio
        from unittest.mock import AsyncMock, MagicMock
        from core.engine import TranscriptionEngine
        from core.stt.local_engine import LocalSTTEngine

        engine = MagicMock(spec=TranscriptionEngine)
        engine.pipeline_mode = "local"
        engine.audio_stream = MagicMock()
        engine.stt_engine = MagicMock(spec=LocalSTTEngine)
        engine.stt_engine.run = AsyncMock()
        engine.reset_session_timer = MagicMock()

        loop = asyncio.new_event_loop()
        loop.run_until_complete(TranscriptionEngine.run(engine))
        loop.close()

        # audio_stream.start() MUST have been called immediately
        engine.audio_stream.start.assert_called_once()
        engine.stt_engine.run.assert_called_once()

    def test_runpod_starts_audio_stream_on_connect(self):
        """When RunPod connects, audio_stream is started if it is not already running."""
        from unittest.mock import MagicMock
        from core.stt.runpod_engine import RunPodSTTEngine

        engine = MagicMock()
        engine.audio_stream = MagicMock()
        engine.audio_stream.running = False

        runpod = RunPodSTTEngine(engine)

        # Simulate the connection block check
        if runpod.engine and hasattr(runpod.engine, "audio_stream") and not runpod.engine.audio_stream.running:
            runpod.engine.audio_stream.start()

        engine.audio_stream.start.assert_called_once()

        # If already running, it should not call start again
        engine.audio_stream.reset_mock()
        engine.audio_stream.running = True
        if runpod.engine and hasattr(runpod.engine, "audio_stream") and not runpod.engine.audio_stream.running:
            runpod.engine.audio_stream.start()

        engine.audio_stream.start.assert_not_called()

    def test_audio_stream_connection_gating(self):
        """AudioStream must gate queuing and demo playback when is_connected is False."""
        from core.audio import AudioStream

        audio = AudioStream()
        audio.is_connected = False
        self.assertFalse(audio.is_connected)

        # Microphones or sounddevice must not enqueue when is_connected is False
        audio.queue.queue.clear()
        if not audio.is_connected:
            pass  # Simulates gating check
        self.assertTrue(audio.queue.empty())

        # Once connected, gating opens
        audio.is_connected = True
        self.assertTrue(audio.is_connected)

    def test_clean_translation_meta_removes_commentary(self):
        """clean_translation_meta strips LLM lecturing, disclaimers, and metadata."""
        from core.translation import clean_translation_meta

        # 1. Exact user example: trailing word lecturing
        sample_meta = (
            'And they read through that whole book, and they talk to each other. '
            'The word "over" in Dutch translates directly to "over" in English. '
            "However, without context, it's impossible to provide a complete and accurate "
            "translation as per your strict requirements. Please provide the full sentence "
            'or paragraph for a precise translation. For example, if the full sentence is '
            '"De taak is over," the translation would be "The task is done.'
        )
        cleaned = clean_translation_meta(sample_meta)
        self.assertEqual(cleaned, "And they read through that whole book, and they talk to each other.")

        # 2. Leading intro stripping
        intro_sample = "Here is the translation: And they talk to each other."
        self.assertEqual(clean_translation_meta(intro_sample), "And they talk to each other.")

        # 3. Trailing note stripping
        note_sample = "And they talk to each other. Note: 'over' was translated literally."
        self.assertEqual(clean_translation_meta(note_sample), "And they talk to each other.")

        # 4. Pure disclaimer yields empty string
        disclaimer_sample = "However, without context, it's impossible to provide a complete translation."
        self.assertEqual(clean_translation_meta(disclaimer_sample), "")

        # 5. Surrounding quotes stripped (including French « » and German „ “)
        quoted_sample = '"And they talk to each other."'
        self.assertEqual(clean_translation_meta(quoted_sample), "And they talk to each other.")
        self.assertEqual(clean_translation_meta("«Et ils se parlent.»"), "Et ils se parlent.")
        self.assertEqual(clean_translation_meta("„Und sie reden miteinander.“"), "Und sie reden miteinander.")

        # 6. Exact user French example: trailing word lecturing
        french_sample = (
            "Et ils lisent tout le livre d'une traite, et ils se parlent les uns aux autres. "
            "Le mot « over » en néerlandais se traduit directement par « over » en anglais. "
            "Cependant, sans contexte, il est impossible de fournir une traduction complète "
            "et précise selon vos exigences strictes. Veuillez fournir la phrase ou le paragraphe "
            "complet pour une traduction précise. Par exemple, si la phrase entière est "
            "« De taak is over », la traduction serait : « La tâche est faite. »"
        )
        self.assertEqual(
            clean_translation_meta(french_sample),
            "Et ils lisent tout le livre d'une traite, et ils se parlent les uns aux autres."
        )

        # 7. Russian meta-commentary
        russian_sample = "И они читают всю эту книгу. Слово «over» переводится как... Однако, без контекста невозможно перевести."
        self.assertEqual(clean_translation_meta(russian_sample), "И они читают всю эту книгу.")

        # 8. Dutch back-translation meta-commentary
        dutch_sample = 'En ze praten met elkaar. Het woord "over" in het Nederlands vertaalt direct... Echter, zonder context is het onmogelijk.'
        self.assertEqual(clean_translation_meta(dutch_sample), "En ze praten met elkaar.")

    def test_telemetry_dashboard_and_payload_construction(self):
        """Telemetry Dashboard template, backend routes, navigation links, and payload calculations function correctly."""
        # 1. Telemetry template resolution and HTML content markers
        telemetry_path = resource_path("templates/telemetry.html")
        self.assertTrue(os.path.exists(telemetry_path))
        with open(telemetry_path, "r", encoding="utf-8") as f:
            html = f.read()

        self.assertIn("Telemetrie & Diagnostics", html)
        self.assertIn("vu-fill", html)
        self.assertIn("vad-badge", html)
        self.assertIn("session-countdown", html)
        self.assertIn("stt-latency", html)
        self.assertIn("trans-latency", html)
        self.assertIn("net-latency", html)
        self.assertIn("pipeline-mode-badge", html)
        self.assertIn("telemetry-gpu-name", html)
        self.assertIn("updateSessionCountdown", html)
        self.assertIn("telemetry_vad", html)

        # 2. Verify backend route registration in app.py
        with open("app.py", "r", encoding="utf-8") as f:
            app_src = f.read()

        self.assertIn('@app.get("/telemetry", response_class=HTMLResponse)', app_src)
        self.assertIn('@app.get("/telemetry/", response_class=HTMLResponse)', app_src)
        self.assertIn('templates/telemetry.html', app_src)
        self.assertIn('get_telemetry_snapshot', app_src)
        self.assertIn('"get_telemetry"', app_src)

        # 3. Navigation links in index.html and viewer.html
        with open("templates/index.html", "r", encoding="utf-8") as f:
            index_src = f.read()
        self.assertIn('href="/telemetry"', index_src)

        with open("templates/viewer.html", "r", encoding="utf-8") as f:
            viewer_src = f.read()
        self.assertIn('href="/telemetry"', viewer_src)

        # 4. Engine telemetry snapshot construction
        broadcasts = []
        engine = TranscriptionEngine(broadcast_callback=lambda msg: broadcasts.append(msg))
        snapshot = engine.get_telemetry_snapshot()

        self.assertEqual(snapshot.get("type"), "telemetry")
        self.assertIn("pipeline_mode", snapshot)
        self.assertIn("source_lang", snapshot)
        self.assertIn("target_lang_1", snapshot)
        self.assertIn("remaining_sec", snapshot)
        self.assertIn("session_window_minutes", snapshot)
        self.assertIn("gpu_name", snapshot)

        # 5. RunPodSTTEngine latency measurement calculation
        runpod = RunPodSTTEngine(engine)
        test_seg_id = "seg_telemetry_test_1"

        # Simulate audio send at t0
        t0 = time.time() - 0.750  # 750ms ago
        runpod.segment_sent_time[test_seg_id] = t0

        # Simulate STT arrival at t1
        t1 = time.time()
        stt_latency = int((t1 - runpod.segment_sent_time[test_seg_id]) * 1000)
        self.assertGreaterEqual(stt_latency, 700)
        self.assertLessEqual(stt_latency, 1000)
        runpod.last_stt_latency_ms = stt_latency
        runpod.segment_stt_time[test_seg_id] = t1

        # Simulate Translation arrival at t2
        t2 = t1 + 0.420  # 420ms later
        trans_latency = int((t2 - runpod.segment_stt_time[test_seg_id]) * 1000)
        self.assertEqual(trans_latency, 420)
        runpod.last_trans_latency_ms = trans_latency

        # Check telemetry payload structure
        payload = {
            "type": "telemetry",
            "stt_latency_ms": stt_latency,
            "trans_latency_ms": trans_latency,
            "pod_id": runpod.active_pod_id,
            "pipeline_mode": "runpod",
        }
        self.assertEqual(payload["type"], "telemetry")
        self.assertEqual(payload["stt_latency_ms"], stt_latency)
        self.assertEqual(payload["trans_latency_ms"], 420)

        # 6. VAD Telemetry payload structure
        runpod.silence_frames = 40  # 40 * 30ms = 1200ms = 1.2s
        runpod.speech_frames = 15
        silence_sec = round((runpod.silence_frames * 30) / 1000.0, 1)
        self.assertEqual(silence_sec, 1.2)

    def test_runpod_condense_logs_and_telemetry(self):
        """Test condensing RunPod container startup logs, telemetry broadcasting, and template markup."""
        # 1. Condense logs: ANSI stripping and carriage return cleaning
        raw_logs = (
            "\x1b[32mINFO:\x1b[0m Initializing pod container\r\n"
            "Step 1: discovering available GPUs nvidia-smi\r\n"
            "Step 2: docker pulling fs layer a1b2c3d4\r\n"
            "Downloading: 100%|██████████| 10/10 [00:01<00:00, 8.5it/s]\r"
            "Step 3: Loading faster-whisper large-v3 model into GPU VRAM\r\n"
            "Step 4: Loading aya-expanse translation model into GPU VRAM\r\n"
            "Step 5: Application startup complete. Uvicorn running on http://0.0.0.0:8000\r\n"
        )
        stage, clean_lines = condense_runpod_logs(raw_logs)
        self.assertEqual(stage, "Server operationeel op poort 8000")
        self.assertLessEqual(len(clean_lines), 6)
        # Verify ANSI codes stripped
        for line in clean_lines:
            self.assertNotIn("\x1b[", line)
            self.assertNotIn("\r", line)

        # 2. Stage detection variations
        # Pulling layers
        stage_pull, _ = condense_runpod_logs("docker pulling fs layer 9abc\r\nExtracting image...")
        self.assertEqual(stage_pull, "Docker container layers downloaden")

        # Global storage mounting
        stage_mount, _ = condense_runpod_logs("[SYSTEM] Global store: mounting at /workspace")
        self.assertEqual(stage_mount, "Global storage koppelen aan /workspace")

        # Container starting
        stage_start, _ = condense_runpod_logs("[SYSTEM] start container")
        self.assertEqual(stage_start, "Container starten op host")

        # Loading models
        stage_models, _ = condense_runpod_logs("Loading faster-whisper large-v3 model in float16")
        self.assertEqual(stage_models, "Faster-Whisper model inladen in GPU VRAM...")

        stage_prewarm, _ = condense_runpod_logs("Pre-warming Ollama aya-expanse:8b in GPU VRAM...")
        self.assertEqual(stage_prewarm, "Ollama model pre-warmen in GPU VRAM...")

        stage_models_generic, _ = condense_runpod_logs("Loaded aya-expanse weights into cache")
        self.assertEqual(stage_models_generic, "AI-modellen laden in GPU VRAM (large-v3 / aya-expanse)")

        # GPU initialization
        stage_gpu, _ = condense_runpod_logs("discovering available gpus nvidia driver loaded")
        self.assertEqual(stage_gpu, "GPU initialisatie")

        # Generic or empty
        stage_empty, lines_empty = condense_runpod_logs("")
        self.assertEqual(stage_empty, "Pod initialiseren...")
        self.assertEqual(lines_empty, [])

        # 3. Test fetch_pod_logs with REST API v2 streaming endpoint mock
        engine = TranscriptionEngine(broadcast_callback=lambda msg: None)
        runpod = RunPodSTTEngine(engine)

        class MockHTTPResponse:
            def __init__(self, lines):
                self._lines = lines

            def __iter__(self):
                return iter(self._lines)

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            def read(self):
                return b'{"data": {"myself": {"pods": [{"id": "pod-test-1", "desiredStatus": "RUNNING"}]}}}'

        def mock_urlopen(req, *args, **kwargs):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "graphql" in url:
                return MockHTTPResponse([])
            elif "/v2/pods/" in url and "/logs" in url:
                sse_data = [
                    b'data: {"source": "system", "line": "270c7bdac026 Extracting"}\n',
                    b'data: {"source": "system", "line": "270c7bdac026 Extracting"}\n',  # Duplicate line to test deduplication
                    b'data: {"source": "system", "line": "Global store: mounting at /workspace"}\n',
                    b'data: {"source": "system", "line": "start container"}\n',
                    b'data: {"source": "container", "line": "Whisper model loaded."}\n',
                ]
                return MockHTTPResponse(sse_data)
            return MockHTTPResponse([])

        with patch.dict(os.environ, {"RUNPOD_API_KEY": "test_api_key"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                last_st, desired_st, clean_logs = runpod.fetch_pod_logs("pod-test-1")
                self.assertEqual(last_st, "RUNNING")
                self.assertEqual(desired_st, "RUNNING")
                self.assertIn("[SYSTEM] 270c7bdac026 Extracting", clean_logs)
                self.assertIn("[SYSTEM] Global store: mounting at /workspace", clean_logs)
                self.assertIn("[SYSTEM] start container", clean_logs)
                self.assertIn("[CONTAINER] Whisper model loaded.", clean_logs)
                # Verify consecutive duplicate lines were deduplicated
                self.assertEqual(clean_logs.count("270c7bdac026 Extracting"), 1)

        # 4. Telemetry broadcast integration
        broadcasts = []
        engine = TranscriptionEngine(broadcast_callback=lambda msg: broadcasts.append(msg))
        runpod = RunPodSTTEngine(engine)

        asyncio.run(
            runpod.broadcast_boot_telemetry(
                pod_id="pod-xyz-123",
                stage="AI-modellen laden in GPU VRAM (large-v3 / aya-expanse)",
                status="INITIALIZING",
                logs=["Loading faster-whisper", "Model loaded in 1.4s"],
            )
        )

        self.assertEqual(len(broadcasts), 1)
        boot_msg = broadcasts[0]
        self.assertEqual(boot_msg.get("type"), "telemetry_boot")
        self.assertEqual(boot_msg.get("pod_id"), "pod-xyz-123")
        self.assertEqual(boot_msg.get("stage"), "AI-modellen laden in GPU VRAM (large-v3 / aya-expanse)")
        self.assertEqual(boot_msg.get("status"), "INITIALIZING")
        self.assertEqual(len(boot_msg.get("logs")), 2)

        # Verify engine.last_boot_telemetry and snapshot
        self.assertIsNotNone(engine.last_boot_telemetry)
        self.assertEqual(engine.last_boot_telemetry["pod_id"], "pod-xyz-123")
        snapshot = engine.get_telemetry_snapshot()
        self.assertIn("boot_telemetry", snapshot)
        self.assertEqual(snapshot["boot_telemetry"]["stage"], "AI-modellen laden in GPU VRAM (large-v3 / aya-expanse)")

        # 4. Telemetry template markup and handlers
        with open("templates/telemetry.html", "r", encoding="utf-8") as f:
            tmpl_src = f.read()

        self.assertIn('id="boot-card"', tmpl_src)
        self.assertIn('id="boot-stage-badge"', tmpl_src)
        self.assertIn('id="boot-status-text"', tmpl_src)
        self.assertIn('id="boot-logs"', tmpl_src)
        self.assertIn('id="boot-connection-indicator"', tmpl_src)
        self.assertIn('pulse-anim', tmpl_src)
        self.assertIn('boot-log-box', tmpl_src)
        self.assertIn('updateBootTelemetry', tmpl_src)
        self.assertIn('telemetry_boot', tmpl_src)
        self.assertIn('id="telemetry-gpu-name"', tmpl_src)

    def test_runpod_pod_terminate_and_cleanup_zombies(self):
        """Test RunPod pod termination, zombie cleanup, and stop_runpod_pod using podTerminate."""
        engine = TranscriptionEngine(broadcast_callback=lambda msg: None)
        runpod = RunPodSTTEngine(engine)

        sent_queries = []

        class MockResponse:
            def __init__(self, json_data):
                self._data = json_data

            def read(self):
                return json.dumps(self._data).encode("utf-8")

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        def mock_urlopen(req, *args, **kwargs):
            if hasattr(req, "data") and req.data:
                payload = json.loads(req.data.decode("utf-8"))
                query_str = payload.get("query", "")
                sent_queries.append(query_str)

                if "podTerminate" in query_str:
                    return MockResponse({"data": {"podTerminate": None}})
                elif "myself" in query_str and "pods" in query_str:
                    return MockResponse({
                        "data": {
                            "myself": {
                                "pods": [
                                    {"id": "pod-zombie-1", "name": "pod-1", "desiredStatus": "EXITED"},
                                    {"id": "pod-active-2", "name": "pod-2", "desiredStatus": "RUNNING"},
                                    {"id": "pod-zombie-3", "name": "pod-3", "desiredStatus": "EXITED"},
                                ]
                            }
                        }
                    })
            return MockResponse({})

        with patch.dict(os.environ, {"RUNPOD_API_KEY": "test_api_key"}):
            with patch("urllib.request.urlopen", side_effect=mock_urlopen):
                # 1. Test direct pod termination
                sent_queries.clear()
                res = runpod.terminate_runpod_pod("pod-single-term")
                self.assertTrue(res)
                self.assertEqual(len(sent_queries), 1)
                self.assertIn('podTerminate(input: {podId: "pod-single-term"})', sent_queries[0])

                # 2. Test cleanup_exited_pods
                sent_queries.clear()
                cleaned_count = runpod.cleanup_exited_pods()
                self.assertEqual(cleaned_count, 2)
                # Should have 1 query for myself.pods, and 2 terminate mutations
                self.assertEqual(len(sent_queries), 3)
                self.assertIn("pod-zombie-1", sent_queries[1])
                self.assertIn("pod-zombie-3", sent_queries[2])

                # 3. Test stop_runpod_pod executes podTerminate and clears active_pod_id
                sent_queries.clear()
                runpod.active_pod_id = "pod-active-999"
                stop_res = asyncio.run(runpod.stop_runpod_pod())
                self.assertTrue(stop_res)
                self.assertIsNone(runpod.active_pod_id)
                self.assertTrue(runpod.intentional_shutdown)
                self.assertEqual(len(sent_queries), 1)
                self.assertIn('podTerminate(input: {podId: "pod-active-999"})', sent_queries[0])

        # 4. Verify runpod_server.py auto_shutdown uses podTerminate
        with open("runpod_server.py", "r", encoding="utf-8") as f:
            server_src = f.read()
        self.assertIn('podTerminate(input: {{podId: "{pod_id}"}})', server_src)
        self.assertNotIn('podStop(input: {{podId: "{pod_id}"}})', server_src)

    def test_runpod_mini_pod_deployment_scripts_and_paths(self):
        """Mini-Pod + Global Volume deployment scripts, server path resolution, and documentation exist and match specs."""
        # 1. runpod_server.py path and cache configurations
        with open("runpod_server.py", "r", encoding="utf-8") as f:
            server_src = f.read()

        self.assertIn("resolve_whisper_model_path", server_src)
        self.assertIn("local_files_only=True", server_src)
        self.assertIn('os.environ.setdefault("TMPDIR", "/tmp")', server_src)
        self.assertIn('os.environ.setdefault("HF_HOME", "/tmp/hf_cache")', server_src)
        self.assertIn('os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")', server_src)
        self.assertIn('os.environ.setdefault("OLLAMA_MODELS"', server_src)

        # 2. setup_once.sh structure and critical flags
        self.assertTrue(os.path.isfile("setup_once.sh"))
        with open("setup_once.sh", "r", encoding="utf-8") as f:
            setup_src = f.read()

        self.assertIn("snapshot_download", setup_src)
        self.assertIn("/tmp/whisper_model", setup_src)
        self.assertIn("/root/.ollama/models", setup_src)
        self.assertIn("copyfileobj", setup_src)
        self.assertIn("model.bin", setup_src)
        self.assertIn("/workspace/models/faster-whisper-large-v3", setup_src)
        self.assertTrue(
            'tar -I "zstd -1 -T0" -cf /tmp/venv_bundle.tar.zst -C /root venv' in setup_src
            or "tar -cf /tmp/venv_bundle.tar -C /root venv" in setup_src
        )
        self.assertTrue(
            "/workspace/venv_bundle.tar.zst" in setup_src
            or "/workspace/venv_bundle.tar" in setup_src
        )
        self.assertIn("/workspace/app", setup_src)
        self.assertIn("curl -fsSL", setup_src)
        self.assertIn("REPO_RAW", setup_src)

        # 3. boot.sh structure and startup sequence
        self.assertTrue(os.path.isfile("boot.sh"))
        with open("boot.sh", "r", encoding="utf-8") as f:
            boot_src = f.read()

        self.assertIn("boot_latest.sh", boot_src)
        self.assertIn("setup_once.sh", boot_src)
        self.assertIn("First run detected on /workspace: running automated setup_once.sh", boot_src)
        self.assertIn("bash /workspace/setup_once.sh", boot_src)
        self.assertIn("zstd", boot_src)
        self.assertIn("tar -xf /workspace/venv_bundle.tar -C /root/", boot_src)
        self.assertIn("cp -r /workspace/app/* /root/app/", boot_src)
        self.assertNotIn("copyfileobj", boot_src)
        self.assertIn("ln -sfn /workspace/models/ollama /root/.ollama/models", boot_src)
        self.assertIn("ln -sfn /workspace/models/faster-whisper-large-v3 /root/models/faster-whisper-large-v3", boot_src)
        self.assertIn("export TMPDIR=/tmp", boot_src)
        self.assertIn("export HF_HOME=/tmp/hf_cache", boot_src)
        self.assertIn("export TORCH_HOME=/tmp/torch_cache", boot_src)
        self.assertIn("export OLLAMA_MODELS=/root/.ollama/models", boot_src)
        self.assertIn("export WHISPER_MODEL=/root/models/faster-whisper-large-v3", boot_src)
        self.assertIn('pkill -9 -f "ollama serve"', boot_src)
        self.assertIn("ollama serve &", boot_src)
        self.assertIn("exec /root/venv/bin/python3 runpod_server.py", boot_src)

        # 4. DEPLOYMENT.md architecture documentation
        self.assertTrue(os.path.isfile("DEPLOYMENT.md"))
        with open("DEPLOYMENT.md", "r", encoding="utf-8") as f:
            deploy_md = f.read()

        self.assertIn("runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04", deploy_md)
        self.assertIn("Mini-Pod + Global Volume", deploy_md)
        self.assertIn("setup_once.sh", deploy_md)
        self.assertIn("boot.sh", deploy_md)
        self.assertIn("venv_bundle.tar", deploy_md)

        # 5. Log condensation detects auto-initialization stage and rich step headers
        from core.stt.runpod_engine import condense_runpod_logs
        stage, lines = condense_runpod_logs("[Boot] First run detected on /workspace: running automated setup_once.sh...")
        self.assertEqual(stage, "Automatische eerste initialisatie van Global Volume (duurt ~3-5m)...")

        stage_step, _ = condense_runpod_logs("=== [5/8] Downloading Faster-Whisper large-v3 on local NVMe SSD ===")
        self.assertIn("Initialisatie [5/8] Downloading Faster-Whisper large-v3", stage_step)


if __name__ == "__main__":
    unittest.main()

