"""Real-world simulation tests for RunPod log parsing, error escalation, and SSE streams."""

import io
import json
import unittest
from unittest.mock import MagicMock, patch

from core.stt.runpod_engine import RunPodSTTEngine, condense_runpod_logs


class TestRealWorldSimulation(unittest.TestCase):
    """Real-world stress and integration simulation for RunPod container log parsing."""

    def test_dirty_log_stream_1000_lines(self):
        """Simulate 1000+ lines of realistic, dirty container output without truncation or failure."""
        lines = []

        # 1. 200 lines of apt-get noise
        for i in range(200):
            lines.append(f"\x1b[32mGet:{i} http://archive.ubuntu.com/ubuntu jammy/main amd64 Packages [1,392 kB]\x1b[0m")
            lines.append(f"Fetched {i*10} kB in 0s (12.4 MB/s)")

        # 2. 500 lines of curl download noise and ANSI progress bars
        for pct in range(1, 101):
            lines.append(f"##O#- # {pct}.0%")
            lines.append(f"### {pct}.0%")
            lines.append(f"\x1b[2K\r {pct}% [=================>] {pct*1000}/100000 12.5MB/s")
            lines.append("######################################################################## 100.0%")
            lines.append("")  # Empty lines

        # 3. Step 7 with Faster-Whisper staging
        lines.append("=== [7/8] Staging Models & Exporting Environment ===")
        lines.append("Staging Faster-Whisper large-v3 from /workspace to local NVMe SSD (/root/models)...")
        lines.append("Copying chunk 1/16")

        raw_logs = "\n".join(lines)
        stage, recent_lines = condense_runpod_logs(raw_logs)

        # Must recognize step [7/8]
        self.assertIn("[7/8]", stage)
        self.assertIn("Faster-Whisper kopiëren naar NVMe SSD", stage)

        # Must not contain curl progress noise in recent lines
        for r_line in recent_lines:
            self.assertFalse(r_line.startswith("##O#"), f"Curl noise found in recent lines: {r_line}")
            self.assertFalse(r_line.startswith("###"), f"Curl noise found in recent lines: {r_line}")

    def test_staging_ollama_models_log_parsing(self):
        """Ensure log parsing for 'Staging Ollama models from /workspace to local NVMe SSD' maps to 'Ollama models kopiëren naar NVMe SSD' in condense_runpod_logs()."""
        raw_logs = """
=== [7/8] Staging Models & Exporting Environment ===
Staging Ollama models from /workspace to local NVMe SSD (/root/.ollama/models)...
Staging model files...
"""
        stage, recent_lines = condense_runpod_logs(raw_logs)
        self.assertIn("[7/8]", stage)
        self.assertIn("Ollama models kopiëren naar NVMe SSD", stage)


    def test_error_pattern_escalation(self):
        """Fatal container errors must be detected immediately and escalated to 'Fout in container' stage."""
        # Scenario A: Explicit ERROR: line
        error_logs_1 = """
=== [1/8] Installing OS Packages ===
Setting up packages...
ERROR: This version requires zstd for extraction
Aborted.
"""
        stage_1, _ = condense_runpod_logs(error_logs_1)
        self.assertTrue(stage_1.startswith("Fout in container:"), f"Expected error escalation, got: {stage_1}")
        self.assertIn("requires zstd for extraction", stage_1)

        # Scenario B: Python Traceback with ModuleNotFoundError
        error_logs_2 = """
=== [8/8] Starting Services ===
Traceback (most recent call last):
  File "runpod_server.py", line 12, in <module>
    import missing_dependency
ModuleNotFoundError: No module named 'missing_dependency'
"""
        stage_2, _ = condense_runpod_logs(error_logs_2)
        self.assertTrue(stage_2.startswith("Fout in container:"), f"Expected error escalation, got: {stage_2}")
        self.assertIn("ModuleNotFoundError", stage_2)

        # Scenario C: Permission Denied or command not found
        error_logs_3 = """
=== [2/8] Installing Ollama ===
bash: /workspace/install.sh: Permission denied: cannot execute
"""
        stage_3, _ = condense_runpod_logs(error_logs_3)
        self.assertTrue(stage_3.startswith("Fout in container:"), f"Expected error escalation, got: {stage_3}")
        self.assertIn("Permission denied", stage_3)

    def test_false_positive_error_detection_prevention(self):
        """Verify benign INFO logs and zero failure metrics do not trigger error badges."""
        # 1. Ollama model recommendations info log with consecutive_failures=0
        info_logs_1 = """
=== [8/8] Starting Services ===
level=INFO source=model_recommendations.go:177 msg="model recommendations cache sleep scheduled" wait=3h consecutive_failures=0
"""
        stage_1, _ = condense_runpod_logs(info_logs_1)
        self.assertFalse(stage_1.startswith("Fout in container:"), f"False-positive error triggered on: {stage_1}")
        self.assertIn("[8/8]", stage_1)

        # 2. Ollama private key generation note
        info_logs_2 = """
=== [8/8] Starting Services ===
Couldn't find '/root/.ollama/id_ed25519'. Generating new private key.
Your new public key is: ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAI...
"""
        stage_2, _ = condense_runpod_logs(info_logs_2)
        self.assertFalse(stage_2.startswith("Fout in container:"), f"False-positive error triggered on: {stage_2}")
        self.assertIn("[8/8]", stage_2)

    def test_all_steps_1_through_8_tracking(self):
        """Verify each phase [1/8] through [8/8] in boot.sh and setup_once.sh is correctly tracked."""
        steps = [
            ("=== [1/8] Installing OS Packages ===", "[1/8]"),
            ("=== [2/8] Installing Ollama ===", "[2/8]"),
            ("=== [3/8] Verifying Ollama Installation in Container OS ===", "[3/8]"),
            ("=== [4/8] Creating Volatile Local SSD Directories ===", "[4/8]"),
            ("=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ===", "[5/8]"),
            ("=== [6/8] Syncing Application Code to /root/app ===", "[6/8]"),
            ("=== [7/8] Staging Models & Exporting Environment ===", "[7/8]"),
            ("=== [8/8] Starting Services ===", "[8/8]"),
        ]

        for header, step_id in steps:
            logs = f"Some prefix noise\n{header}\nRunning sub-process..."
            stage, _ = condense_runpod_logs(logs)
            self.assertIn(step_id, stage, f"Failed to detect step {step_id} in {stage}")

    def test_sse_stream_parser_and_status_derivation(self):
        """Simulate RunPod SSE streaming endpoint (/v2/pods/{id}/logs) with mixed formats and corrupt JSON."""
        sse_lines = [
            b"data: {\"source\":\"system\",\"line\":\"Discovering available GPUs...\"}\n",
            b"data: {\"source\":\"system\",\"line\":\"Discovering available GPUs...\"}\n",  # Duplicate to deduplicate
            b"data: {\"source\":\"container\",\"line\":\"=== [3/8] Verifying Ollama Installation in Container OS ===\"}\n",
            b"data: {\"source\":\"container\",\"line\":\"Restoring Ollama engine from Global Volume cache (1-2s)...\"}\n",
            b"data: invalid json payload here\n",
            b": keepalive ping\n",
            b"Raw non-SSE container stdout line\n",
        ]

        # Mock urllib.request.urlopen returning our sse_lines iterator
        mock_response = MagicMock()
        mock_response.__iter__.return_value = iter(sse_lines)
        mock_response.__enter__.return_value = mock_response

        # Mock GraphQL pod query
        graphql_response_data = json.dumps({
            "data": {
                "myself": {
                    "pods": [
                        {"id": "pod-sim-123", "desiredStatus": "RUNNING"}
                    ]
                }
            }
        }).encode("utf-8")
        mock_graphql_resp = MagicMock()
        mock_graphql_resp.read.return_value = graphql_response_data
        mock_graphql_resp.__enter__.return_value = mock_graphql_resp

        engine = RunPodSTTEngine()

        def mock_urlopen(req, timeout=None):
            url = req.full_url if hasattr(req, "full_url") else str(req)
            if "graphql" in url:
                return mock_graphql_resp
            return mock_response

        with patch("urllib.request.urlopen", side_effect=mock_urlopen):
            derived_status, desired_status, logs_text = engine.fetch_pod_logs("pod-sim-123", "test_key")

        self.assertEqual(desired_status, "RUNNING")
        self.assertIn("[3/8]", logs_text)
        self.assertIn("Restoring Ollama engine", logs_text)
        self.assertIn("Raw non-SSE container stdout line", logs_text)

        # Stage and status derived correctly
        self.assertIn("Initialisatie [3/8]", engine.condense_runpod_logs(logs_text)[0])

    def test_telemetry_continues_during_connection_refused(self):
        """Verify telemetry broadcast and log polling continue actively while WebSocket connection refuses."""
        import asyncio

        mock_engine = MagicMock()
        mock_engine.broadcast = MagicMock()

        stt_engine = RunPodSTTEngine(engine=mock_engine)
        stt_engine.active_pod_id = "pod-refuse-test"

        # Mock fetch_pod_logs returning progress lines
        stt_engine.fetch_pod_logs = MagicMock(return_value=(
            "RUNNING", "RUNNING",
            "=== [5/8] Extracting Python Virtual Environment to NVMe SSD (/root/venv) ===\nExtracting..."
        ))

        async def run_test():
            attempts = 0

            class MockConnectCM:
                def __init__(self, *args, **kwargs):
                    pass

                async def __aenter__(self):
                    nonlocal attempts
                    attempts += 1
                    if attempts <= 2:
                        raise ConnectionRefusedError("Connection refused on port 8000")
                    raise asyncio.CancelledError()

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    pass

            mock_ws_module = MagicMock()
            mock_ws_module.connect = MockConnectCM

            async def mock_sleep(duration):
                pass

            with patch("core.stt.runpod_engine.websockets", mock_ws_module), \
                 patch("asyncio.sleep", side_effect=mock_sleep):
                try:
                    await stt_engine.run()
                except asyncio.CancelledError:
                    pass

        asyncio.run(run_test())

        # Verify broadcast was called during connection retries
        self.assertGreaterEqual(mock_engine.broadcast.call_count, 2)
        # Verify the stage contains [5/8]
        found_step_5 = False
        for call in mock_engine.broadcast.call_args_list:
            args, kwargs = call
            payload = args[0] if args else kwargs.get("data", {})
            if "[5/8]" in str(payload):
                found_step_5 = True
                break
        self.assertTrue(found_step_5, "Telemetry did not broadcast step [5/8] during connection retries")

    def test_interim_translation_debouncing_and_backtranslation_gating(self):
        """Simulate translation chain skipping of interim tasks during busy state and guaranteed sequential final completion."""
        import asyncio

        sent_messages = []
        mock_websocket = MagicMock()
        async def mock_send_json(payload):
            sent_messages.append(payload)
        mock_websocket.send_json = mock_send_json

        config = {
            "source_lang": "nl",
            "target_lang_1": "en",
            "target_lang_2": "ru",
            "target_lang_3": "nl",
            "target_lang_4": "nl",
            "enable_trans_3": True,
            "enable_trans_4": True,
        }

        async def mock_translate_text(text, target, source):
            if "interim_1" in text:
                # Simulate translation taking 50ms
                await asyncio.sleep(0.05)
            return f"[{target}:{text}]"

        active_chain_tasks = {}
        chain_busy = {"trans1": False, "trans2": False}

        async def run_simulation():
            # Helper replicating runpod_server.py dispatch logic
            async def process_translation_chain(col_key_1, lang_1, col_key_2, lang_2, segment_id, segment_status, current_text, prev_task):
                if prev_task and not prev_task.done():
                    try:
                        await prev_task
                    except Exception:
                        pass
                chain_busy[col_key_1] = True
                try:
                    text_1 = await mock_translate_text(current_text, lang_1, config["source_lang"])
                    col_idx_1 = "col2_text" if col_key_1 == "trans1" else "col3_text"
                    await mock_websocket.send_json({
                        "id": segment_id,
                        "status": segment_status,
                        "is_delta": False,
                        col_idx_1: text_1
                    })
                    if col_key_2 and lang_2 and segment_status == "final":
                        text_2 = await mock_translate_text(text_1, lang_2, lang_1)
                        col_idx_2 = "col4_text" if col_key_2 == "trans3" else "col5_text"
                        await mock_websocket.send_json({
                            "id": segment_id,
                            "status": segment_status,
                            "is_delta": False,
                            col_idx_2: text_2
                        })
                finally:
                    chain_busy[col_key_1] = False

            def dispatch_segment(seg_id, status, text):
                c3_key = "trans3" if config.get("enable_trans_3") else None
                c3_lang = config.get("target_lang_3") if config.get("enable_trans_3") else None
                if chain_busy.get("trans1", False):
                    if status == "final":
                        prev = active_chain_tasks.get("trans1")
                        active_chain_tasks["trans1"] = asyncio.create_task(
                            process_translation_chain("trans1", config.get("target_lang_1"), c3_key, c3_lang, seg_id, status, text, prev)
                        )
                    # else: skip interim chunk without cancelling HTTP requests!
                else:
                    chain_busy["trans1"] = True
                    prev = active_chain_tasks.get("trans1")
                    active_chain_tasks["trans1"] = asyncio.create_task(
                        process_translation_chain("trans1", config.get("target_lang_1"), c3_key, c3_lang, seg_id, status, text, prev)
                    )

            # 1. Dispatch first interim chunk (busy for 50ms)
            dispatch_segment("seg_1", "interim", "interim_1 audio")
            await asyncio.sleep(0.01)

            # 2. Dispatch second interim chunk while trans1 is busy (must be cleanly skipped)
            dispatch_segment("seg_1", "interim", "interim_2 audio")
            await asyncio.sleep(0.01)

            # 3. Dispatch final chunk while trans1 is still finishing chunk 1 (must wait and then run)
            dispatch_segment("seg_1", "final", "final audio")

            # Wait for all queued tasks to finish cleanly
            while any(not t.done() for t in active_chain_tasks.values()):
                await asyncio.sleep(0.01)

        asyncio.run(run_simulation())

        # Verify interim_1 finished cleanly without being cancelled
        sent_texts = [m.get("col2_text") for m in sent_messages if "col2_text" in m]
        self.assertIn("[en:interim_1 audio]", sent_texts)
        # Verify interim_2 was skipped to prevent Ollama socket contention
        self.assertNotIn("[en:interim_2 audio]", sent_texts)
        # Verify final was executed sequentially and guaranteed
        self.assertIn("[en:final audio]", sent_texts)

        # Verify back-translation was ONLY sent for final status, never for interim
        col4_messages = [m for m in sent_messages if "col4_text" in m]
        self.assertEqual(len(col4_messages), 1)
        self.assertEqual(col4_messages[0]["status"], "final")
        self.assertEqual(col4_messages[0]["col4_text"], "[nl:[en:final audio]]")

    def test_interim_hallucination_does_not_suppress_subsequent_chunks(self):
        """Verify that an interim chunk producing a hallucination ('***') does NOT suppress subsequent speech and translations."""
        import asyncio
        import time

        broadcast_events = []
        mock_engine = MagicMock()
        async def mock_broadcast(msg):
            broadcast_events.append(msg)
        mock_engine.broadcast = mock_broadcast
        mock_engine.source_lang = "nl"
        mock_engine.target_lang_1 = "en"
        mock_engine.target_lang_2 = "ru"
        mock_engine.target_lang_3 = "nl"
        mock_engine.target_lang_4 = "nl"
        mock_engine.enable_trans_3 = True
        mock_engine.enable_trans_4 = True
        mock_engine.restart_required = False
        mock_engine.is_paused = False
        mock_engine.session_expiry_time = time.time() + 3600
        mock_engine.initial_session_window_minutes = 15
        mock_engine.glossary_phrases = None
        mock_audio_stream = MagicMock()
        mock_audio_stream.running = True
        mock_audio_stream.read_chunk.return_value = None
        mock_audio_stream.calculate_rms.return_value = 0.0
        mock_audio_stream.is_speech.return_value = False
        mock_engine.audio_stream = mock_audio_stream

        stt_engine = RunPodSTTEngine(engine=mock_engine)
        stt_engine.active_pod_id = "pod-hallucination-test"
        stt_engine.discover_runpod_url = MagicMock(return_value="ws://127.0.0.1:8000/stream")

        messages = [
            # 1. First interim chunk generates hallucination '***'
            json.dumps({"id": "seg_100", "status": "interim", "col1_text": "***"}),
            # 2. Second interim chunk contains legitimate partial speech
            json.dumps({"id": "seg_100", "status": "interim", "col1_text": "Neem die vraag mee"}),
            # 3. Final chunk contains full speech
            json.dumps({"id": "seg_100", "status": "final", "col1_text": "Neem die vraag mee. En dat zijn de woorden fundament."}),
            # 4. Final translation message
            json.dumps({"id": "seg_100", "status": "final", "col2_text": "Take that question with you. And those are the words foundation."}),
        ]

        class MockWS:
            def __init__(self, msgs):
                self.msgs = msgs

            async def send(self, data):
                pass

            def __aiter__(self):
                self._iter = iter(self.msgs)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    mock_engine.restart_required = True
                    await asyncio.sleep(0.01)
                    raise asyncio.CancelledError()

        connected = False
        class MockConnectCM:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                nonlocal connected
                if connected:
                    raise asyncio.CancelledError()
                connected = True
                return MockWS(messages)

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_ws_module = MagicMock()
        mock_ws_module.connect = MockConnectCM

        with patch("core.stt.runpod_engine.websockets", mock_ws_module):
            async def run_with_timeout():
                try:
                    await asyncio.wait_for(stt_engine.run(), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            asyncio.run(run_with_timeout())

        # 1. Segment must NOT be suppressed
        self.assertNotIn("seg_100", stt_engine.suppressed_seg_ids)

        # 2. Verify broadcast received the legitimate speech and translation
        col1_broadcasts = [e["col1_text"] for e in broadcast_events if "col1_text" in e]
        self.assertIn("Neem die vraag mee", col1_broadcasts)
        self.assertIn("Neem die vraag mee. En dat zijn de woorden fundament.", col1_broadcasts)
        self.assertNotIn("***", col1_broadcasts)

        col2_broadcasts = [e["col2_text"] for e in broadcast_events if "col2_text" in e]
        self.assertIn("Take that question with you. And those are the words foundation.", col2_broadcasts)

        # 3. Verify genuine final hallucination suppression
        mock_engine.restart_required = False
        stt_engine_final = RunPodSTTEngine(engine=mock_engine)
        stt_engine_final.active_pod_id = "pod-hallucination-final-test"
        stt_engine_final.discover_runpod_url = MagicMock(return_value="ws://127.0.0.1:8000/stream")

        final_messages = [
            json.dumps({"id": "seg_fake", "status": "final", "col1_text": "Ondertiteling: TV Gelderland"}),
            json.dumps({"id": "seg_fake", "status": "final", "col2_text": "Subtitles: TV Gelderland"}),
        ]

        class MockFinalWS:
            def __init__(self, msgs):
                self.msgs = msgs

            async def send(self, data):
                pass

            def __aiter__(self):
                self._iter = iter(self.msgs)
                return self

            async def __anext__(self):
                try:
                    return next(self._iter)
                except StopIteration:
                    mock_engine.restart_required = True
                    await asyncio.sleep(0.01)
                    raise asyncio.CancelledError()

        final_connected = False
        class MockFinalConnectCM:
            def __init__(self, *args, **kwargs):
                pass

            async def __aenter__(self):
                nonlocal final_connected
                if final_connected:
                    raise asyncio.CancelledError()
                final_connected = True
                return MockFinalWS(final_messages)

            async def __aexit__(self, exc_type, exc_val, exc_tb):
                pass

        mock_ws_final = MagicMock()
        mock_ws_final.connect = MockFinalConnectCM

        broadcast_events.clear()
        with patch("core.stt.runpod_engine.websockets", mock_ws_final):
            async def run_final_with_timeout():
                try:
                    await asyncio.wait_for(stt_engine_final.run(), timeout=0.1)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            asyncio.run(run_final_with_timeout())

        self.assertIn("seg_fake", stt_engine_final.suppressed_seg_ids)
        subtitle_events = [e for e in broadcast_events if "col1_text" in e or "col2_text" in e]
        self.assertEqual(len(subtitle_events), 0)


if __name__ == "__main__":
    unittest.main()

