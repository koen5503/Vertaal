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


if __name__ == "__main__":
    unittest.main()

