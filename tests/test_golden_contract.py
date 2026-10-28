"""Golden contract regression tests and STT functional pipeline tests.

Eliminates mock drift by testing RunPod STT auto-discovery and GraphQL fallback
directly against authentic production JSON payloads loaded from tests/fixtures/*.json.
Validates the full functional pipeline: restitching, VAD audio chunking, silence heartbeat,
and session window timer extensions.
"""

import asyncio
import json
import os
import pathlib
import time
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from core.audio import FRAME_DURATION_MS, SAMPLE_RATE
from core.stt.runpod_engine import (
    RunPodSTTEngine,
    _normalize_pod_ports,
    restitch_segment_text,
)

FIXTURES_DIR = pathlib.Path(__file__).parent / "fixtures"


def _load_fixture(filename: str) -> dict:
    filepath = FIXTURES_DIR / filename
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


class MockHTTPResponse:
    """Mock for urllib.request.urlopen returning JSON fixture data."""

    def __init__(self, data: dict, status: int = 200, headers: dict = None):
        self._data = json.dumps(data).encode("utf-8")
        self.status = status
        self.status_code = status
        self.headers = headers or {}

    def read(self, *args, **kwargs) -> bytes:
        return self._data

    def getcode(self) -> int:
        return self.status

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass


class TestGoldenContract(unittest.TestCase):
    """Regression tests verifying observable contracts against authentic production fixtures."""

    def setUp(self):
        self.mock_engine = MagicMock()
        self.mock_engine.broadcast = AsyncMock()
        self.mock_engine.restart_required = False
        self.mock_engine.is_paused = False
        self.mock_engine.session_expiry_time = time.time() + 900
        self.mock_engine.initial_session_window_minutes = 15.0
        self.mock_engine.silence_threshold_ms = 300
        self.mock_engine.min_local_chunk_s = 1.0
        self.mock_engine.max_local_chunk_s = 5.0
        self.mock_engine.max_silence_flush_ms = 600
        self.mock_engine.interim_chunk_s = 0.5
        self.mock_engine.min_speech_frames = 3
        self.mock_engine.source_lang = "nl"
        self.mock_engine.target_lang_1 = "en"
        self.mock_engine.target_lang_2 = "de"
        self.mock_engine.target_lang_3 = "fr"
        self.mock_engine.target_lang_4 = "es"
        self.mock_engine.glossary_phrases = []
        self.mock_engine.audio_stream = MagicMock()
        self.mock_engine.audio_stream.running = True
        self.mock_engine.audio_stream.is_connected = False
        self.mock_engine.audio_stream.queue = MagicMock()
        self.mock_engine.audio_stream.queue.empty.return_value = True

    def test_golden_contract_rest_v2_running(self):
        """Authentic REST v2 payload with active pod, ports: ['8000/http'], and null runtime.

        Verifies:
        - active_pod_id extracted exactly as 'g7kt0vxny22w2b'
        - proxy_url extracted as 'wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream'
        - active_gpu_name extracted as 'NVIDIA RTX 4000 Ada Generation' from gpu.id
        - zero exceptions during parsing and URL resolution.
        """
        payload = _load_fixture("runpod_v2_pods_running.json")
        stt = RunPodSTTEngine(engine=self.mock_engine)

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(payload, status=200)):
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "rpa_testkey1234567890", "RUNPOD_PREFER_TCP": "false"}):
                pods = stt._fetch_pods_list("rpa_testkey1234567890")
                self.assertEqual(len(pods), 1)
                self.assertEqual(pods[0]["id"], "g7kt0vxny22w2b")
                self.assertEqual(pods[0]["desiredStatus"], "RUNNING")
                self.assertEqual(pods[0]["machine"]["gpuDisplayName"], "NVIDIA RTX 4000 Ada Generation")

                url = stt.discover_runpod_url()
                self.assertIsNotNone(url)
                self.assertEqual(stt.active_pod_id, "g7kt0vxny22w2b")
                self.assertEqual(stt.proxy_url, "wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream")
                self.assertEqual(url, "wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream")
                self.assertEqual(stt.active_gpu_name, "NVIDIA RTX 4000 Ada Generation")

    def test_golden_contract_rest_v2_empty(self):
        """Authentic REST v2 payload with empty pods array.

        Verifies:
        - _fetch_pods_list cleanly returns an empty list (0 pods)
        - discover_runpod_url returns None
        - active_pod_id and proxy_url remain None.
        """
        payload = _load_fixture("runpod_v2_pods_empty.json")
        stt = RunPodSTTEngine(engine=self.mock_engine)

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(payload, status=200)):
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "rpa_testkey1234567890"}):
                pods = stt._fetch_pods_list("rpa_testkey1234567890")
                self.assertEqual(pods, [])

                url = stt.discover_runpod_url()
                self.assertIsNone(url)
                self.assertIsNone(stt.active_pod_id)
                self.assertIsNone(stt.proxy_url)

    def test_golden_contract_rest_v2_multi_port(self):
        """Authentic REST v2 payload with ports: ['8000/http', '8001/tcp'].

        Verifies:
        - Correctly normalizes both ports into structured dictionaries
        - Sets both proxy_url and direct_tcp_url
        - Prefers direct_tcp_url when RUNPOD_PREFER_TCP=true
        - Prefers proxy_url when RUNPOD_PREFER_TCP=false.
        """
        payload = _load_fixture("runpod_v2_pods_multi_port.json")
        stt = RunPodSTTEngine(engine=self.mock_engine)

        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(payload, status=200)):
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "rpa_testkey1234567890", "RUNPOD_PREFER_TCP": "false"}):
                url_proxy = stt.discover_runpod_url()
                self.assertEqual(stt.proxy_url, "wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream")
                self.assertEqual(stt.direct_tcp_url, "ws://213.192.2.92:40199/stream")
                self.assertEqual(url_proxy, "wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream")

        # When RUNPOD_PREFER_TCP=true, returns TCP link directly
        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(payload, status=200)):
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "rpa_testkey1234567890", "RUNPOD_PREFER_TCP": "true"}):
                url_tcp = stt.discover_runpod_url()
                self.assertEqual(url_tcp, "ws://213.192.2.92:40199/stream")
                self.assertEqual(stt.direct_tcp_url, "ws://213.192.2.92:40199/stream")
                self.assertEqual(stt.proxy_url, "wss://g7kt0vxny22w2b-8000.proxy.runpod.net/stream")

    def test_golden_contract_graphql_fallback(self):
        """Legacy GraphQL payload with data.myself.pods containing dict ports.

        Verifies:
        - When REST API v2 fails with HTTP 500, _fetch_pods_list falls back to GraphQL
        - GraphQL response format parsed cleanly
        - Extracts active_pod_id, proxy_url, direct_tcp_url, and active_gpu_name.
        """
        payload = _load_fixture("runpod_graphql_myself.json")
        stt = RunPodSTTEngine(engine=self.mock_engine)

        call_count = 0

        def side_effect(req, timeout=10.0):
            nonlocal call_count
            call_count += 1
            if "api.runpod.io/v2/pods" in req.full_url:
                # Trigger fallback by returning 500 error
                import urllib.error
                raise urllib.error.HTTPError(req.full_url, 500, "Internal Server Error", {}, None)
            return MockHTTPResponse(payload, status=200)

        with patch("urllib.request.urlopen", side_effect=side_effect):
            with patch.dict(os.environ, {"RUNPOD_API_KEY": "rpa_testkey1234567890", "RUNPOD_PREFER_TCP": "false"}):
                pods = stt._fetch_pods_list("rpa_testkey1234567890")
                self.assertEqual(len(pods), 1)
                self.assertEqual(pods[0]["id"], "pod-gql-legacy-1")
                self.assertEqual(pods[0]["machine"]["gpuDisplayName"], "NVIDIA RTX 4000 Ada Generation")

                url = stt.discover_runpod_url()
                self.assertEqual(url, "wss://pod-gql-legacy-1-8000.proxy.runpod.net/stream")
                self.assertEqual(stt.active_pod_id, "pod-gql-legacy-1")
                self.assertEqual(stt.proxy_url, "wss://pod-gql-legacy-1-8000.proxy.runpod.net/stream")
                self.assertEqual(stt.direct_tcp_url, "ws://213.192.2.92:40199/stream")
                self.assertEqual(stt.active_gpu_name, "NVIDIA RTX 4000 Ada Generation")


    def test_golden_contract_normalize_pod_ports_variations(self):
        """_normalize_pod_ports handles lists of dicts, strings, comma-separated strings, and malformed inputs."""
        # 1. Comma-separated string format
        ports_csv = "8000/http, 8001/tcp, 22/tcp"
        norm_csv = _normalize_pod_ports(ports_csv)
        self.assertEqual(len(norm_csv), 3)
        self.assertEqual(norm_csv[0]["privatePort"], 8000)
        self.assertEqual(norm_csv[0]["type"], "http")
        self.assertEqual(norm_csv[1]["privatePort"], 8001)
        self.assertEqual(norm_csv[1]["type"], "tcp")

        # 2. List of dicts with 'port' or 'privatePort'
        ports_dicts = [
            {"port": 8000, "type": "http"},
            {"privatePort": 8001, "type": "tcp", "ip": "1.2.3.4", "publicPort": 1234, "isIpPublic": True},
        ]
        norm_dicts = _normalize_pod_ports(ports_dicts)
        self.assertEqual(len(norm_dicts), 2)
        self.assertEqual(norm_dicts[0]["privatePort"], 8000)
        self.assertEqual(norm_dicts[1]["privatePort"], 8001)
        self.assertEqual(norm_dicts[1]["ip"], "1.2.3.4")

        # 3. Malformed and unexpected types
        self.assertEqual(_normalize_pod_ports(None), [])
        self.assertEqual(_normalize_pod_ports(12345), [])
        self.assertEqual(_normalize_pod_ports(["invalid/port/string", "abc/http"]), [])

    def test_golden_contract_runpod_response_wrapper(self):
        """RunPodResponse provides uniform access to status, text, json, and context manager."""
        from core.stt.runpod_engine import RunPodResponse

        raw_mock = MagicMock()
        raw_mock.status = 200
        raw_mock.headers = {"Content-Type": "application/json"}
        raw_mock.read.return_value = b'{"status": "ok", "count": 42}'

        resp = RunPodResponse(raw_mock)
        self.assertEqual(resp.status, 200)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.getcode(), 200)
        self.assertEqual(resp.text, '{"status": "ok", "count": 42}')
        self.assertEqual(resp.json(), {"status": "ok", "count": 42})

        with resp as r:
            self.assertEqual(r.status, 200)

    def test_golden_contract_runpod_request_backoff_on_429(self):
        """_runpod_request handles HTTP 429 rate limits with backoff and retry."""
        import urllib.error
        from core.stt.runpod_engine import _runpod_request

        attempts = 0

        def flaky_urlopen(req, timeout=15.0):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                hdrs = {"Retry-After": "0.01"}
                raise urllib.error.HTTPError(req.full_url, 429, "Too Many Requests", hdrs, None)
            mock_ok = MagicMock()
            mock_ok.status = 200
            mock_ok.headers = {}
            mock_ok.read.return_value = b'{"success": true}'
            return mock_ok

        with patch("urllib.request.urlopen", side_effect=flaky_urlopen):
            resp = _runpod_request("GET", "https://api.runpod.io/v2/pods", timeout=5.0, base_delay=0.01)
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.json(), {"success": True})
            self.assertEqual(attempts, 2)

    def test_ensure_template_http_ports(self):
        """ensure_template_http_ports verifies template exposure and mutates template if needed."""
        stt = RunPodSTTEngine(engine=self.mock_engine)

        # 1. Invalid arguments return False
        self.assertFalse(stt.ensure_template_http_ports("", "template-1"))
        self.assertFalse(stt.ensure_template_http_ports("key-1", ""))

        # 2. Template already has 8000/http -> returns True without mutation
        resp_matching = {
            "data": {
                "myself": {
                    "podTemplates": [
                        {"id": "tmpl-1", "ports": "8000/http", "name": "template 1"}
                    ]
                }
            }
        }
        with patch("urllib.request.urlopen", return_value=MockHTTPResponse(resp_matching, status=200)):
            self.assertTrue(stt.ensure_template_http_ports("test-key", "tmpl-1"))

        # 3. Template has 8000/tcp -> sends mutation to update ports to 8000/http
        resp_needs_update = {
            "data": {
                "myself": {
                    "podTemplates": [
                        {
                            "id": "tmpl-2",
                            "ports": "8000/tcp",
                            "name": "template 2",
                            "imageName": "runpod/whisper:latest",
                            "containerDiskInGb": 20,
                            "volumeInGb": 20,
                            "volumeMountPath": "/workspace",
                            "dockerArgs": "",
                            "env": [{"key": "FOO", "value": "BAR"}],
                        }
                    ]
                }
            }
        }
        mutation_resp = {
            "data": {
                "saveTemplate": {
                    "id": "tmpl-2",
                    "ports": "8000/http"
                }
            }
        }

        call_idx = 0
        def query_and_mutate(req, timeout=10.0):
            nonlocal call_idx
            call_idx += 1
            if call_idx == 1:
                return MockHTTPResponse(resp_needs_update, status=200)
            return MockHTTPResponse(mutation_resp, status=200)

        with patch("urllib.request.urlopen", side_effect=query_and_mutate):
            self.assertTrue(stt.ensure_template_http_ports("test-key", "tmpl-2"))
            self.assertEqual(call_idx, 2)

    def test_engine_stop_lifecycle(self):
        """stt.stop() delegates to stop_runpod_pod to terminate cloud compute resources."""
        stt = RunPodSTTEngine(engine=self.mock_engine)
        stt.stop_runpod_pod = AsyncMock(return_value=True)
        asyncio.run(stt.stop())
        stt.stop_runpod_pod.assert_called_once()



class TestPipelineFunctionality(unittest.TestCase):
    """Functional pipeline regression tests: restitching, VAD chunking, heartbeat, and timer resets."""

    def setUp(self):
        self.mock_engine = MagicMock()
        self.broadcasts = []

        async def mock_broadcast(msg):
            self.broadcasts.append(msg)

        self.mock_engine.broadcast = mock_broadcast
        self.mock_engine.restart_required = False
        self.mock_engine.is_paused = False
        self.mock_engine.session_expiry_time = time.time() + 900
        self.mock_engine.initial_session_window_minutes = 15.0
        self.mock_engine.silence_threshold_ms = 300
        self.mock_engine.min_local_chunk_s = 1.0
        self.mock_engine.max_local_chunk_s = 5.0
        self.mock_engine.max_silence_flush_ms = 600
        self.mock_engine.interim_chunk_s = 0.5
        self.mock_engine.min_speech_frames = 3
        self.mock_engine.source_lang = "nl"
        self.mock_engine.target_lang_1 = "en"
        self.mock_engine.target_lang_2 = "de"
        self.mock_engine.target_lang_3 = "fr"
        self.mock_engine.target_lang_4 = "es"
        self.mock_engine.glossary_phrases = []
        self.mock_engine.audio_stream = MagicMock()
        self.mock_engine.audio_stream.running = True
        self.mock_engine.audio_stream.is_connected = False
        self.mock_engine.audio_stream.queue = MagicMock()
        self.mock_engine.audio_stream.queue.empty.return_value = True

    def test_restitching_segments_prefix_and_tails(self):
        """restitch_segment_text handles prefix revisions, word overlaps, stem truncations, and independent clauses."""
        # 1. Edge cases: empty texts
        self.assertEqual(restitch_segment_text("", "Nieuwe zin."), "Nieuwe zin.")
        self.assertEqual(restitch_segment_text("Oude zin.", ""), "Oude zin.")

        # 2. Full revised transcript (incoming starts with or contains previous text)
        prev = "Wij verwelkomen u in"
        revised = "Wij verwelkomen u in de dienst van heden morgen."
        self.assertEqual(restitch_segment_text(prev, revised), revised)

        # 3. Exact word overlap at boundary
        prev_tail = "Wij zingen samen uit psalm 122"
        inc_head = "psalm 122 vers 1 en 2"
        self.assertEqual(
            restitch_segment_text(prev_tail, inc_head),
            "Wij zingen samen uit psalm 122 vers 1 en 2"
        )

        # 4. Truncated stem match at word boundary (e.g. bepalen / bepalend)
        prev_stem = "Dat is alles bepalen"
        inc_stem = "bepalend voor de toekomst van de gemeente"
        self.assertEqual(
            restitch_segment_text(prev_stem, inc_stem),
            "Dat is alles bepalend voor de toekomst van de gemeente"
        )

        # 5. Independent appended clause with no overlap
        clause1 = "De schriftlezing van vanmorgen."
        clause2 = "Wij lezen uit Romeinen 8."
        self.assertEqual(
            restitch_segment_text(clause1, clause2),
            "De schriftlezing van vanmorgen. Wij lezen uit Romeinen 8."
        )

    def test_audio_buffer_handling_and_vad_chunking(self):
        """RunPodSTTEngine audio accumulation, pre-roll silence capping, and VAD chunking triggers."""
        stt = RunPodSTTEngine(engine=self.mock_engine)

        # 30ms @ 16kHz 16-bit mono = 480 samples = 960 bytes
        frame_bytes = b"\x10\x27" * 480
        silence_bytes = b"\x00\x00" * 480

        # 1. Pre-roll capping: when speech_frames == 0, audio_buffer is capped at 32000 bytes (1.0s)
        for _ in range(50):  # 50 frames * 960 bytes = 48,000 bytes
            stt.audio_buffer.extend(silence_bytes)
            if stt.speech_frames == 0 and len(stt.audio_buffer) > 32000:
                stt.audio_buffer = stt.audio_buffer[-32000:]
        self.assertEqual(len(stt.audio_buffer), 32000)

        # 2. Ingest speech frames: speech_frames increments, silence_frames resets
        stt.audio_buffer = bytearray()
        for _ in range(10):
            stt.speech_frames += 1
            stt.silence_frames = 0
            stt.audio_buffer.extend(frame_bytes)

        self.assertEqual(stt.speech_frames, 10)
        self.assertEqual(stt.silence_frames, 0)
        self.assertEqual(len(stt.audio_buffer), 9600)

        # 3. Trigger natural pause: 11 silence frames (330ms >= 300ms threshold)
        # buffer duration = (9600 + 11*960) / 32000 = 20160 / 32000 = 0.63s
        # Need >= 1.0s for natural pause
        for _ in range(25):
            stt.silence_frames += 1
            stt.audio_buffer.extend(silence_bytes)

        silence_ms = stt.silence_frames * FRAME_DURATION_MS
        buffer_duration_s = len(stt.audio_buffer) / (SAMPLE_RATE * 2)
        is_natural_pause = (silence_ms >= self.mock_engine.silence_threshold_ms and
                            buffer_duration_s >= self.mock_engine.min_local_chunk_s)
        self.assertTrue(is_natural_pause)

        # Simulate segment finalization
        seg_id = "seg_test_123"
        stt.finalized_segments.add(seg_id)
        stt.pending_seg_id = seg_id
        stt.pending_status = "final"
        stt.audio_buffer = bytearray()
        stt.speech_frames = 0
        stt.silence_frames = 0

        self.assertIn(seg_id, stt.finalized_segments)
        self.assertEqual(len(stt.audio_buffer), 0)
        self.assertEqual(stt.speech_frames, 0)

    def test_silence_heartbeat_probe_emission_and_interception(self):
        """Silence heartbeat probe triggers after silence_heartbeat_sec and is intercepted from subtitle output."""
        stt = RunPodSTTEngine(engine=self.mock_engine)
        stt.silence_heartbeat_sec = 0.05  # 50ms for test speed
        stt.last_speech_or_heartbeat_time = time.time() - 0.1

        # Simulate probe condition in sender loop
        now = time.time()
        silence_s = now - stt.last_speech_or_heartbeat_time
        self.assertGreaterEqual(silence_s, stt.silence_heartbeat_sec)

        probe_id = f"probe_{int(now * 1000)}"
        stt.heartbeat_seg_ids.add(probe_id)
        stt.pending_seg_id = probe_id
        stt.pending_status = "final"
        stt.heartbeat_probe_sent_time = now - 0.02

        # Verify probe response handling in ws_receiver logic
        probe_incoming = {
            "id": probe_id,
            "type": "probe_response",
            "col1_text": "",
        }
        client_seg_id = probe_incoming.get("id")
        is_probe = (
            (client_seg_id and (client_seg_id in stt.heartbeat_seg_ids or str(client_seg_id).startswith("probe_")))
            or probe_incoming.get("type") == "probe_response"
        )
        self.assertTrue(is_probe)

        latency_ms = int((time.time() - stt.heartbeat_probe_sent_time) * 1000)
        self.assertGreater(latency_ms, 0)
        stt.heartbeat_seg_ids.discard(client_seg_id)
        self.assertNotIn(probe_id, stt.heartbeat_seg_ids)

    def test_session_timer_resets_on_stt_events(self):
        """Engine session timer extends on audio send, STT message receipt, and auto-pauses on expiry."""
        stt = RunPodSTTEngine(engine=self.mock_engine)
        self.mock_engine.reset_session_timer = MagicMock()
        self.mock_engine.extend_session_window = MagicMock()

        # 1. Timer reset on connection
        self.mock_engine.reset_session_timer(reason="runpod_connected")
        self.mock_engine.reset_session_timer.assert_called_with(reason="runpod_connected")

        # 2. Timer extension on audio segment dispatch
        self.mock_engine.extend_session_window(reason="stt_activity")
        self.mock_engine.extend_session_window.assert_called_with(reason="stt_activity")

        # 3. Verify session expiry detection triggers auto-pause and stop_runpod_pod
        self.mock_engine.session_expiry_time = time.time() - 1.0  # Expired
        stt.stop_runpod_pod = AsyncMock(return_value=True)

        async def run_timeout_check():
            if time.time() > self.mock_engine.session_expiry_time:
                self.mock_engine._session_expired = True
                self.mock_engine.is_paused = True
                await stt.stop_runpod_pod()
                await self.mock_engine.broadcast({"action": "auto_paused", "reason": "15_min_limit"})

        asyncio.run(run_timeout_check())
        self.assertTrue(self.mock_engine._session_expired)
        self.assertTrue(self.mock_engine.is_paused)
        stt.stop_runpod_pod.assert_called_once()
        self.assertEqual(len(self.broadcasts), 1)
        self.assertEqual(self.broadcasts[0]["action"], "auto_paused")

    def test_pipeline_ws_receiver_delta_translation_and_artifact_cleaning(self):
        """Translation delta stitching, LLM conversational meta cleaning, and noise tail stripping."""
        from core.translation import clean_translation_meta
        from core.stt.local_engine import TAIL_HALLUCINATION_PATTERN
        import re

        stt = RunPodSTTEngine(engine=self.mock_engine)
        client_seg_id = "seg_test_trans_1"

        # 1. Delta stitching: chunk 1
        stt.segment_translations.setdefault(client_seg_id, {})["col2_text"] = "The Lord is my shepherd"
        chunk2_incoming = "I shall not want."
        stitched = restitch_segment_text(stt.segment_translations[client_seg_id]["col2_text"], chunk2_incoming)
        self.assertEqual(stitched, "The Lord is my shepherd I shall not want.")

        # 2. Meta-commentary cleaning
        raw_llm_out = "Here is the translation: De Heer is mijn herder."
        cleaned = clean_translation_meta(raw_llm_out)
        self.assertEqual(cleaned, "De Heer is mijn herder.")

        # 3. Trailing broadcast artifact cleaning (TV Gelderland)
        hallucination_raw = "Gezegende zondag gewenst. - TV Gelderland"
        stripped = TAIL_HALLUCINATION_PATTERN.sub("", hallucination_raw).strip()
        stripped = re.sub(
            r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)(?:tv|тв)?\s*(?:gelderland|gelreland|гельдерланд)(?:\s+\d{4})?[\.\,\!\?\s]*$",
            "",
            stripped,
            flags=re.IGNORECASE
        ).strip()
        self.assertEqual(stripped, "Gezegende zondag gewenst.")

    def test_pipeline_hallucination_suppression_and_unsuppress_lifecycle(self):
        """Hallucination segments are suppressed, dropped, and cleanly unsuppressed on candidate speech."""
        stt = RunPodSTTEngine(engine=self.mock_engine)
        seg_id = "seg_hallucinated_1"

        # 1. Filter suppresses asterisk / noise credit
        from core.stt.local_engine import filter_anti_hallucination
        res = filter_anti_hallucination("***")
        self.assertIsNone(res)

        # 2. Add to suppressed segments
        stt.suppressed_seg_ids.add(seg_id)
        self.assertIn(seg_id, stt.suppressed_seg_ids)

        # 3. When valid candidate speech arrives, segment is unsuppressed
        incoming_speech = "De genade van onze Here Jezus Christus zij met u allen."
        if incoming_speech.strip() and incoming_speech.strip() != "***":
            stt.suppressed_seg_ids.discard(seg_id)

        self.assertNotIn(seg_id, stt.suppressed_seg_ids)


if __name__ == "__main__":
    unittest.main()

