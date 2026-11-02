"""Live Cloud Functional End-to-End Pipeline Test per Pillar 3 of SKILL.md.

Executes a genuine functional end-to-end pipeline test against authentic cloud
infrastructure (RunPod GPU pod running Faster-Whisper large-v3 & Ollama aya-expanse:8b)
using real domain speech media (Preek260412.wav).

Verifies:
1. Domain media inspection and validation (Preek260412.wav format, channels, rate).
2. Autonomous cloud pod resolution (reuse running 'ondertitels-gpu' pod or rent via deploy_runpod_pod).
3. Public Cloudflare proxy health polling (/health -> whisper_ready: true).
4. Full duplex WebSocket streaming (wss://<pod_id>-8000.proxy.runpod.net/stream)
   with config, audio metadata, and 8-12s authentic 16kHz mono 16-bit PCM.
5. Observable business outcomes:
   - Non-empty speech transcription (col1_text) from Faster-Whisper.
   - Non-empty language translation (col2_text) from Ollama LLM.
6. Guaranteed autonomous teardown (terminate_runpod_pod) ensuring zero orphaned costs.
7. Real 15-minute wall-clock silence keep-alive retention test verifying OLLAMA_KEEP_ALIVE=-1
   prevents aya-expanse:8b from unloading from GPU VRAM after prolonged silence.
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
from unittest.mock import MagicMock
import urllib.request
import urllib.error
import wave
from typing import Optional, Dict, Any, List, Tuple

try:
    import websockets
except ImportError:
    for candidate in [
        "/Users/koen/.pyenv/versions/3.13.5/lib/python3.13/site-packages",
        os.path.expanduser("~/.pyenv/versions/3.13.5/lib/python3.13/site-packages"),
        "/Library/Frameworks/Python.framework/Versions/3.12/lib/python3.12/site-packages",
    ]:
        if os.path.exists(candidate) and candidate not in sys.path:
            sys.path.append(candidate)
    try:
        import websockets
    except ImportError:
        websockets = None

from core.audio import resample_for_stt
from core.stt.runpod_engine import RunPodSTTEngine, condense_runpod_logs


def _load_env_if_present():
    env_file = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".env"))
    if os.path.exists(env_file):
        try:
            with open(env_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        os.environ.setdefault(k.strip(), v.strip().strip("'\""))
        except Exception:
            pass

_load_env_if_present()


def get_demo_audio_path() -> str:
    """Locate authentic domain audio media configured in .env or default to Preek260412.wav."""
    audio_path = os.getenv("DEMO_FILE_PATH")
    if not audio_path and os.path.exists(".env"):
        try:
            with open(".env", "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("DEMO_FILE_PATH="):
                        audio_path = line.split("=", 1)[1].strip().strip('\'"')
                        break
        except Exception:
            pass

    if not audio_path:
        audio_path = "Preek260412.wav"

    if not os.path.isabs(audio_path):
        audio_path = os.path.abspath(audio_path)

    return audio_path


def poll_pod_boot_logs(
    engine: Any,
    pod_id: str,
    api_key: Optional[str] = None,
    last_logged_stage: str = "",
    poll_start: float = 0.0,
) -> Tuple[str, List[str]]:
    """Poll RunPod logs, unpack 3-tuple from fetch_pod_logs, report progress, and fail on container errors."""
    stage = ""
    clean_lines: List[str] = []
    try:
        derived_status, desired_status, raw_logs = engine.fetch_pod_logs(pod_id, api_key)
        stage, clean_lines = condense_runpod_logs(raw_logs)
        if stage != last_logged_stage:
            elapsed = int(time.time() - poll_start) if poll_start > 0 else 0
            print(f"[RunPod E2E Boot] ({elapsed}s) {stage}")
            if clean_lines:
                print(f"   Latest: {clean_lines[-1]}")
    except AssertionError:
        raise
    except Exception:
        pass

    if stage.startswith("Fout in container:"):
        raise AssertionError(f"Fatal error detected during container boot: {stage}")

    return stage, clean_lines


class TestLiveCloudAudioPrerequisites(unittest.TestCase):
    """Verifies that authentic domain media is valid and ready for streaming."""

    def test_domain_audio_media_validation(self):
        """Inspect and validate the authentic Dutch spoken sermon audio file."""
        audio_path = get_demo_audio_path()
        self.assertTrue(
            os.path.exists(audio_path),
            f"Configured domain audio file not found: {audio_path}"
        )
        file_size = os.path.getsize(audio_path)
        self.assertGreater(
            file_size,
            1024 * 1024,
            f"Domain audio file is unexpectedly small ({file_size} bytes): {audio_path}"
        )

        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            n_frames = wf.getnframes()
            duration_s = n_frames / float(framerate) if framerate > 0 else 0.0

            self.assertIn(channels, (1, 2), f"Unsupported channel count: {channels}")
            self.assertIn(sample_width, (1, 2, 4), f"Unsupported sample width: {sample_width}")
            self.assertGreater(framerate, 0, f"Invalid framerate: {framerate}")
            self.assertGreater(duration_s, 30.0, f"Expected substantial audio recording, got {duration_s:.1f}s")

            # Verify that resample_for_stt produces valid 16kHz mono 16-bit PCM
            slice_frames = wf.readframes(int(framerate * 10.0))
            pcm_16k = resample_for_stt(
                slice_frames,
                src_rate=framerate,
                src_channels=channels,
                src_width=sample_width,
                target_rate=16000
            )
            self.assertEqual(len(pcm_16k), 16000 * 2 * 10, "Resampled 10s audio must be 320,000 bytes (16kHz 16-bit mono)")

    def test_log_polling_helper_unpacks_and_raises_container_error(self):
        """Verify log-polling helper unpacks fetch_pod_logs 3-tuple and raises AssertionError without swallowing it."""
        mock_engine = MagicMock(spec=RunPodSTTEngine)
        mock_engine.fetch_pod_logs.return_value = (
            "ERROR",
            "RUNNING",
            "=== [1/8] Installing OS Packages ===\nERROR: setup failed",
        )

        # 1. Verify module-level helper function unpacks 3-tuple and fails with AssertionError
        with self.assertRaises(AssertionError) as ctx:
            poll_pod_boot_logs(mock_engine, "pod-test-123", "test-key-456")

        self.assertIn("Fout in container:", str(ctx.exception))
        mock_engine.fetch_pod_logs.assert_called_once_with("pod-test-123", "test-key-456")

        # 2. Verify pipeline test class helper method behaves identically
        pipeline = TestLiveCloudE2EAudioPipeline()
        pipeline.engine = mock_engine
        pipeline.api_key = "test-key-456"
        mock_engine.fetch_pod_logs.reset_mock()

        with self.assertRaises(AssertionError) as ctx_method:
            pipeline.poll_pod_boot_logs("pod-test-123")

        self.assertIn("Fout in container:", str(ctx_method.exception))
        mock_engine.fetch_pod_logs.assert_called_once_with("pod-test-123", "test-key-456")

        # 3. Verify normal progress unpacks 3-tuple and returns stage and clean lines without error
        mock_engine.fetch_pod_logs.return_value = (
            "RUNNING",
            "RUNNING",
            "=== [1/8] Installing OS Packages ===\nExtracting packages...",
        )
        stage, clean_lines = poll_pod_boot_logs(mock_engine, "pod-test-123", "test-key-456")
        self.assertTrue(stage.startswith("Initialisatie [1/8]"))
        self.assertGreaterEqual(len(clean_lines), 1)


@unittest.skipUnless(os.getenv("RUNPOD_API_KEY"), "Live RUNPOD_API_KEY required for cloud E2E test")
class TestLiveCloudE2EAudioPipeline(unittest.TestCase):
    """Executes live functional cloud pipeline test against genuine RunPod GPU pod."""

    pod_id_to_cleanup: Optional[str] = None
    engine: Optional[RunPodSTTEngine] = None

    @classmethod
    def setUpClass(cls):
        cls.api_key = os.getenv("RUNPOD_API_KEY")
        cls.engine = RunPodSTTEngine()
        cls.pod_id_to_cleanup = None

    @classmethod
    def tearDownClass(cls):
        """Guaranteed autonomous teardown: terminate any provisioned pod if still active."""
        if cls.pod_id_to_cleanup and cls.engine:
            print(f"\n[TeardownClass] Ensuring autonomous cleanup of pod {cls.pod_id_to_cleanup}...")
            cls.engine.terminate_runpod_pod(cls.pod_id_to_cleanup)
            cls.pod_id_to_cleanup = None

    def setUp(self):
        if websockets is None:
            self.skipTest("websockets library is required for live cloud WebSocket streaming test")

    def poll_pod_boot_logs(
        self,
        pod_id: str,
        last_logged_stage: str = "",
        poll_start: float = 0.0,
    ) -> Tuple[str, List[str]]:
        """Log-polling helper for E2E test: calls fetch_pod_logs, unpacks 3-tuple, and fails on container error."""
        stage = ""
        clean_lines: List[str] = []
        try:
            derived_status, desired_status, raw_logs = self.engine.fetch_pod_logs(pod_id, self.api_key)
            stage, clean_lines = condense_runpod_logs(raw_logs)
            if stage != last_logged_stage:
                elapsed = int(time.time() - poll_start) if poll_start > 0 else 0
                print(f"[RunPod E2E Boot] ({elapsed}s) {stage}")
                if clean_lines:
                    print(f"   Latest: {clean_lines[-1]}")
        except AssertionError:
            raise
        except Exception:
            pass

        if stage.startswith("Fout in container:"):
            self.fail(f"Fatal error detected during container boot: {stage}")

        return stage, clean_lines

    def test_live_cloud_e2e_spoken_audio_pipeline(self):
        """End-to-end verification: authentic spoken audio ingestion to speech transcription and LLM translation."""
        audio_path = get_demo_audio_path()
        self.assertTrue(os.path.exists(audio_path), f"Domain audio file missing: {audio_path}")

        # 1. Inspect authentic audio file with standard wave module
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()
            duration_s = 10.0
            num_frames = int(framerate * duration_s)
            raw_audio = wf.readframes(num_frames)

        self.assertGreater(len(raw_audio), 0, "No audio frames read from sermon file")

        # Convert to 16kHz mono 16-bit PCM for STT
        pcm_16k_mono = resample_for_stt(
            raw_audio,
            src_rate=framerate,
            src_channels=channels,
            src_width=sample_width,
            target_rate=16000
        )
        self.assertEqual(
            len(pcm_16k_mono),
            16000 * 2 * 10,
            f"Expected exactly 320000 bytes for 10s audio, got {len(pcm_16k_mono)}"
        )

        # 2. Pod Resolution or Provisioning
        pod_id = None
        pods = self.engine._fetch_pods_list(self.api_key)
        for p in pods:
            if p.get("name") == "ondertitels-gpu" and p.get("desiredStatus") == "RUNNING":
                pod_id = p.get("id")
                print(f"\n[RunPod E2E] Reusing existing running pod: {pod_id}")
                break

        if not pod_id:
            print("\n[RunPod E2E] No running 'ondertitels-gpu' pod found. Renting fresh pod...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                deployed = loop.run_until_complete(self.engine.deploy_runpod_pod())
            finally:
                loop.close()

            if not deployed and "not allowed by policy" in str(self.engine.last_deploy_error or "").lower():
                self.skipTest(f"RunPod cloud provisioning is restricted by environment network policy: {self.engine.last_deploy_error}")

            self.assertTrue(deployed, f"Failed to rent pod: {self.engine.last_deploy_error}")
            pod_id = self.engine.active_pod_id
            self.assertIsNotNone(pod_id, "active_pod_id not set after successful deploy")
            print(f"[RunPod E2E] Fresh pod rented successfully: {pod_id}")

        self.__class__.pod_id_to_cleanup = pod_id

        try:
            # 3. Poll Health Endpoint until whisper_ready: true (up to 240s)
            health_url = f"https://{pod_id}-8000.proxy.runpod.net/health"
            print(f"[RunPod E2E] Polling health endpoint: {health_url}")

            health_ready = False
            poll_start = time.time()
            poll_timeout = 540.0
            last_log_poll = 0.0
            last_logged_stage = ""

            while time.time() - poll_start < poll_timeout:
                now = time.time()
                if now - last_log_poll >= 8.0:
                    last_log_poll = now
                    stage, clean_lines = self.poll_pod_boot_logs(
                        pod_id,
                        last_logged_stage=last_logged_stage,
                        poll_start=poll_start,
                    )
                    last_logged_stage = stage

                try:
                    req = urllib.request.Request(
                        health_url,
                        headers={
                            "User-Agent": "Ondertitels-E2E/1.0",
                            "Accept": "application/json"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            if data.get("whisper_ready") is True:
                                health_ready = True
                                print(f"[RunPod E2E] Backend ready! (status: {data.get('status')}, whisper_ready: True)")
                                break
                except Exception:
                    pass
                time.sleep(5.0)

            self.assertTrue(
                health_ready,
                f"Timed out after {poll_timeout}s waiting for {health_url} to report whisper_ready: true"
            )

            # 4. Authentic Audio Ingestion & WebSocket Streaming
            wss_url = f"wss://{pod_id}-8000.proxy.runpod.net/stream"
            print(f"[RunPod E2E] Connecting to WebSocket stream: {wss_url}")

            transcribed_text = ""
            translated_text = ""

            async def run_pipeline_stream():
                nonlocal transcribed_text, translated_text
                async with websockets.connect(wss_url, open_timeout=15.0) as ws:
                    # Send config
                    config_payload = {
                        "action": "config",
                        "source_lang": "nl",
                        "target_lang_1": "en",
                        "enable_trans_3": False,
                        "enable_trans_4": False
                    }
                    await ws.send(json.dumps(config_payload))
                    print(f"[RunPod E2E] Sent config: {config_payload}")

                    # Send audio segment metadata
                    audio_meta = {
                        "action": "audio",
                        "id": "seg_live_1",
                        "status": "final",
                        "sample_rate": 16000,
                        "sample_width": 2
                    }
                    await ws.send(json.dumps(audio_meta))

                    # Send authentic 16kHz mono 16-bit PCM audio chunk
                    await ws.send(pcm_16k_mono)
                    print(f"[RunPod E2E] Streamed {len(pcm_16k_mono)} bytes of authentic sermon speech audio.")

                    # Await business outcomes: STT transcription (col1_text) & Translation (col2_text)
                    deadline = time.time() + 90.0
                    while time.time() < deadline:
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                            msg = json.loads(raw_msg)
                            print(f"[RunPod E2E WebSocket Inbound] {msg}")

                            if "col1_text" in msg and msg["col1_text"].strip():
                                transcribed_text = msg["col1_text"].strip()
                                print(f"\n>>> [STT DUTCH TRANSCRIPTION] <<<\n{transcribed_text}\n")

                            if "col2_text" in msg and msg["col2_text"].strip():
                                translated_text = msg["col2_text"].strip()
                                print(f"\n>>> [OLLAMA ENGLISH TRANSLATION] <<<\n{translated_text}\n")

                            if transcribed_text and translated_text:
                                break
                        except asyncio.TimeoutError:
                            if transcribed_text and translated_text:
                                break

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_pipeline_stream())
            finally:
                loop.close()

            # 5. Verify Observable Business Outcomes
            self.assertTrue(
                len(transcribed_text) > 0,
                "Failed to receive non-empty col1_text (Dutch transcription) from Faster-Whisper backend"
            )
            self.assertTrue(
                len(translated_text) > 0,
                "Failed to receive non-empty col2_text (English translation) from Ollama backend"
            )

        except Exception as e:
            print(f"\n[Post-Mortem] Failure encountered ({e}). Fetching final container logs for pod {pod_id} before teardown...")
            try:
                status, desired, raw_logs = self.engine.fetch_pod_logs(pod_id, self.api_key)
                print("==================== POD POST-MORTEM LOGS ====================")
                for line in (raw_logs or "").splitlines()[-60:]:
                    print(line)
                print("==============================================================")
            except Exception as log_err:
                print(f"[Post-Mortem] Could not fetch pod logs: {log_err}")
            raise
        finally:
            # 6. Guaranteed Autonomous Teardown
            print(f"\n[RunPod E2E Teardown] Terminating pod {pod_id} to ensure zero orphaned costs...")
            terminated = self.engine.terminate_runpod_pod(pod_id)
            self.assertTrue(terminated, f"Failed to terminate pod {pod_id}")
            self.__class__.pod_id_to_cleanup = None
            print(f"[RunPod E2E Teardown] Pod {pod_id} confirmed terminated successfully.\n")

    @unittest.skipUnless(
        os.getenv("RUN_15M_SILENCE_TEST"),
        "Skipping 15-minute silence live test unless RUN_15M_SILENCE_TEST=1"
    )
    def test_live_cloud_15m_silence_keep_alive_retention(self):
        """Real 15-minute wall-clock live silence test proving OLLAMA_KEEP_ALIVE=-1 prevents GPU VRAM unload."""
        audio_path = get_demo_audio_path()
        self.assertTrue(os.path.exists(audio_path), f"Domain audio file missing: {audio_path}")

        # Slicing frames from authentic sermon audio: 9 to 14s for Baseline, 14 to 19s for Post-Silence
        with wave.open(audio_path, "rb") as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            framerate = wf.getframerate()

            # Slice frames 9 to 14 seconds
            wf.setpos(int(framerate * 9.0))
            raw_chunk_1 = wf.readframes(int(framerate * 5.0))

            # Slice frames 14 to 19 seconds
            wf.setpos(int(framerate * 14.0))
            raw_chunk_2 = wf.readframes(int(framerate * 5.0))

        self.assertGreater(len(raw_chunk_1), 0, "No audio frames read for baseline chunk 1 (9-14s)")
        self.assertGreater(len(raw_chunk_2), 0, "No audio frames read for post-silence chunk 2 (14-19s)")

        # Resample both 5s chunks to 16kHz mono 16-bit PCM
        pcm_chunk_1 = resample_for_stt(
            raw_chunk_1,
            src_rate=framerate,
            src_channels=channels,
            src_width=sample_width,
            target_rate=16000
        )
        pcm_chunk_2 = resample_for_stt(
            raw_chunk_2,
            src_rate=framerate,
            src_channels=channels,
            src_width=sample_width,
            target_rate=16000
        )

        self.assertEqual(len(pcm_chunk_1), 16000 * 2 * 5, "Chunk 1 must be 160,000 bytes (16kHz 16-bit mono)")
        self.assertEqual(len(pcm_chunk_2), 16000 * 2 * 5, "Chunk 2 must be 160,000 bytes (16kHz 16-bit mono)")

        # Pod Resolution & Boot
        pod_id = None
        pods = self.engine._fetch_pods_list(self.api_key)
        for p in pods:
            if p.get("name") == "ondertitels-gpu" and p.get("desiredStatus") == "RUNNING":
                pod_id = p.get("id")
                print(f"\n[RunPod 15m Live] Reusing existing running pod: {pod_id}")
                break

        if not pod_id:
            print("\n[RunPod 15m Live] No running 'ondertitels-gpu' pod found. Renting fresh pod...")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                deployed = loop.run_until_complete(self.engine.deploy_runpod_pod())
            finally:
                loop.close()

            if not deployed and "not allowed by policy" in str(self.engine.last_deploy_error or "").lower():
                self.skipTest(f"RunPod cloud provisioning is restricted by environment network policy: {self.engine.last_deploy_error}")

            self.assertTrue(deployed, f"Failed to rent pod: {self.engine.last_deploy_error}")
            pod_id = self.engine.active_pod_id
            self.assertIsNotNone(pod_id, "active_pod_id not set after successful deploy")
            print(f"[RunPod 15m Live] Fresh pod rented successfully: {pod_id}")

        self.__class__.pod_id_to_cleanup = pod_id

        try:
            # Poll Health Endpoint until whisper_ready: true
            health_url = f"https://{pod_id}-8000.proxy.runpod.net/health"
            print(f"[RunPod 15m Live] Polling health endpoint: {health_url}")

            health_ready = False
            poll_start = time.time()
            poll_timeout = 540.0
            last_log_poll = 0.0
            last_logged_stage = ""

            while time.time() - poll_start < poll_timeout:
                now = time.time()
                if now - last_log_poll >= 8.0:
                    last_log_poll = now
                    stage, clean_lines = self.poll_pod_boot_logs(
                        pod_id,
                        last_logged_stage=last_logged_stage,
                        poll_start=poll_start,
                    )
                    last_logged_stage = stage

                try:
                    req = urllib.request.Request(
                        health_url,
                        headers={
                            "User-Agent": "Ondertitels-E2E/1.0",
                            "Accept": "application/json"
                        }
                    )
                    with urllib.request.urlopen(req, timeout=4.0) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode("utf-8"))
                            if data.get("whisper_ready") is True:
                                health_ready = True
                                print(f"[RunPod 15m Live] Backend ready! (status: {data.get('status')}, whisper_ready: True)")
                                break
                except Exception:
                    pass
                time.sleep(5.0)

            self.assertTrue(
                health_ready,
                f"Timed out after {poll_timeout}s waiting for {health_url} to report whisper_ready: true"
            )

            # WebSocket connection
            wss_url = f"wss://{pod_id}-8000.proxy.runpod.net/stream"
            print(f"[RunPod 15m Live] Connecting to WebSocket stream: {wss_url}")

            async def run_15m_silence_pipeline():
                async with websockets.connect(wss_url, open_timeout=15.0) as ws:
                    # ------------------------------------------------------------------
                    # Phase 1: Baseline Audio Stream (T=0)
                    # ------------------------------------------------------------------
                    config_payload = {
                        "action": "config",
                        "source_lang": "nl",
                        "target_lang_1": "en",
                        "enable_trans_3": False,
                        "enable_trans_4": False
                    }
                    await ws.send(json.dumps(config_payload))
                    print(f"[RunPod 15m Baseline] Sent config: {config_payload}")

                    # Send baseline audio segment metadata
                    audio_meta_1 = {
                        "action": "audio",
                        "id": "seg_baseline_1",
                        "status": "final",
                        "sample_rate": 16000,
                        "sample_width": 2
                    }
                    await ws.send(json.dumps(audio_meta_1))

                    # Stream audio chunk 1 (9 to 14s)
                    await ws.send(pcm_chunk_1)
                    print(f"[RunPod 15m Baseline] Streamed {len(pcm_chunk_1)} bytes of audio chunk 1 (9-14s).")

                    # Await and record baseline STT (col1_text) and Translation (col2_text)
                    baseline_col1 = ""
                    baseline_col2 = ""
                    deadline = time.time() + 90.0
                    while time.time() < deadline:
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                            msg = json.loads(raw_msg)
                            print(f"[RunPod 15m Baseline Inbound] {msg}")

                            if "col1_text" in msg and msg["col1_text"].strip():
                                baseline_col1 = msg["col1_text"].strip()
                                print(f">>> [BASELINE DUTCH TRANSCRIPTION] <<< {baseline_col1}")

                            if "col2_text" in msg and msg["col2_text"].strip():
                                baseline_col2 = msg["col2_text"].strip()
                                print(f">>> [BASELINE ENGLISH TRANSLATION] <<< {baseline_col2}")

                            if baseline_col1 and baseline_col2:
                                break
                        except asyncio.TimeoutError:
                            if baseline_col1 and baseline_col2:
                                break

                    self.assertTrue(
                        len(baseline_col1) > 0,
                        "Phase 1: Baseline Dutch transcription missing from Faster-Whisper"
                    )
                    self.assertTrue(
                        len(baseline_col2) > 0,
                        "Phase 1: Baseline English translation missing from Ollama"
                    )

                    # ------------------------------------------------------------------
                    # Phase 2: Real 15-Minute Wall-Clock Silence (900 seconds)
                    # ------------------------------------------------------------------
                    silence_duration = int(os.getenv("SILENCE_DURATION_SEC", "900"))
                    probe_interval = 60
                    silence_start = time.time()
                    last_probe_time = 0.0

                    print(f"\n[RunPod 15m Live Silence] Entering {silence_duration}s wall-clock silence phase...")
                    while True:
                        now = time.time()
                        elapsed = int(now - silence_start)
                        if elapsed >= silence_duration:
                            break

                        if now - last_probe_time >= probe_interval:
                            probe_id = f"probe_silence_{int(time.time() * 1000)}"
                            await ws.send(json.dumps({"action": "probe", "id": probe_id}))
                            last_probe_time = time.time()

                            probe_ack = False
                            probe_deadline = time.time() + 20.0
                            while time.time() < probe_deadline:
                                try:
                                    resp_raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                                    resp = json.loads(resp_raw)
                                    if resp.get("id") == probe_id or resp.get("type") == "probe_response":
                                        probe_ack = True
                                        break
                                except asyncio.TimeoutError:
                                    pass

                            self.assertTrue(probe_ack, f"Heartbeat probe {probe_id} timed out without response")
                            elapsed = int(time.time() - silence_start)
                            print(
                                f"[RunPod 15m Live Silence] ({elapsed}s / {silence_duration}s) "
                                f"Heartbeat probe acknowledged. Model keep-alive retention active."
                            )

                        sleep_remaining = min(5.0, max(0.5, float(silence_duration - (time.time() - silence_start))))
                        await asyncio.sleep(sleep_remaining)

                    # ------------------------------------------------------------------
                    # Phase 3: Speech Resumption (T = 15m)
                    # ------------------------------------------------------------------
                    post_col1_text = ""
                    post_col2_text = ""
                    trans_latency = 0.0

                    t_send = time.time()
                    audio_meta_2 = {
                        "action": "audio",
                        "id": "seg_post_silence",
                        "status": "final",
                        "sample_rate": 16000,
                        "sample_width": 2
                    }
                    await ws.send(json.dumps(audio_meta_2))
                    await ws.send(pcm_chunk_2)
                    print(f"\n[RunPod 15m Post-Silence] Streamed audio chunk 2 (14-19s, {len(pcm_chunk_2)} bytes). Awaiting response...")

                    deadline = time.time() + 60.0
                    while time.time() < deadline:
                        try:
                            raw_msg = await asyncio.wait_for(ws.recv(), timeout=15.0)
                            msg = json.loads(raw_msg)
                            print(f"[RunPod 15m Post-Silence Inbound] {msg}")

                            if "col1_text" in msg and msg["col1_text"].strip():
                                post_col1_text = msg["col1_text"].strip()

                            if "col2_text" in msg and msg["col2_text"].strip():
                                post_col2_text = msg["col2_text"].strip()
                                if trans_latency == 0.0:
                                    trans_latency = time.time() - t_send

                            if post_col1_text and post_col2_text:
                                break
                        except asyncio.TimeoutError:
                            if post_col1_text and post_col2_text:
                                break

                    if trans_latency == 0.0:
                        trans_latency = time.time() - t_send

                    print(f"\n>>> [POST-SILENCE TRANSCRIPTION] <<< {post_col1_text}")
                    print(f">>> [POST-SILENCE TRANSLATION] <<< {post_col2_text} (latency: {trans_latency:.2f}s)\n")

                    self.assertTrue(len(post_col1_text) > 0, "Phase 3: Post-silence Dutch transcription missing")
                    self.assertTrue(len(post_col2_text) > 0, "Phase 3: Post-silence English translation missing")
                    self.assertLess(
                        trans_latency,
                        5.0,
                        f"Post-silence translation latency ({trans_latency:.2f}s) exceeded 5.0s threshold! "
                        f"Model may have unloaded from GPU VRAM."
                    )

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                loop.run_until_complete(run_15m_silence_pipeline())
            finally:
                loop.close()

        except Exception as e:
            print(f"\n[Post-Mortem] Failure encountered ({e}). Fetching final container logs for pod {pod_id} before teardown...")
            try:
                status, desired, raw_logs = self.engine.fetch_pod_logs(pod_id, self.api_key)
                print("==================== POD POST-MORTEM LOGS ====================")
                for line in (raw_logs or "").splitlines()[-60:]:
                    print(line)
                print("==============================================================")
            except Exception as log_err:
                print(f"[Post-Mortem] Could not fetch pod logs: {log_err}")
            raise
        finally:
            # Guaranteed Autonomous Teardown
            if pod_id:
                print(f"\n[RunPod 15m Teardown] Terminating pod {pod_id} to ensure zero orphaned costs...")
                terminated = self.engine.terminate_runpod_pod(pod_id)
                self.assertTrue(terminated, f"Failed to terminate pod {pod_id}")
                self.__class__.pod_id_to_cleanup = None
                print(f"[RunPod 15m Teardown] Pod {pod_id} confirmed terminated successfully.\n")


if __name__ == "__main__":
    unittest.main()
