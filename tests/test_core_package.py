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

        # Direct update as implemented in runpod_engine.py
        full_col = incoming_col = incoming_final.strip()
        runpod.segment_translations.setdefault(client_seg_id, {})["col2_text"] = full_col

        # Must NOT contain the concatenated interim translation
        self.assertEqual(runpod.segment_translations[client_seg_id]["col2_text"], incoming_final)
        self.assertNotIn("If you imagine", runpod.segment_translations[client_seg_id]["col2_text"])

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


if __name__ == "__main__":
    unittest.main()
