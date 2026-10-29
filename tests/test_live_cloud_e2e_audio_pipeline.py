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
"""

import asyncio
import json
import logging
import os
import sys
import time
import unittest
import urllib.request
import urllib.error
import wave
from typing import Optional, Dict, Any, List

try:
    import websockets
except ImportError:
    websockets = None

from core.audio import resample_for_stt
from core.stt.runpod_engine import RunPodSTTEngine


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

            self.assertTrue(deployed, f"Failed to rent pod: {self.engine.last_deploy_error}")
            pod_id = self.engine.active_pod_id
            self.assertIsNotNone(pod_id, "active_pod_id not set after successful deploy")
            print(f"[RunPod E2E] Fresh pod rented successfully: {pod_id}")

        self.__class__.pod_id_to_cleanup = pod_id

        # 3. Poll Health Endpoint until whisper_ready: true (up to 240s)
        health_url = f"https://{pod_id}-8000.proxy.runpod.net/health"
        print(f"[RunPod E2E] Polling health endpoint: {health_url}")

        health_ready = False
        poll_start = time.time()
        poll_timeout = 240.0

        while time.time() - poll_start < poll_timeout:
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
            except Exception as e:
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
                deadline = time.time() + 60.0
                while time.time() < deadline:
                    try:
                        raw_msg = await asyncio.wait_for(ws.recv(), timeout=12.0)
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

        try:
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

        finally:
            # 6. Guaranteed Autonomous Teardown
            print(f"\n[RunPod E2E Teardown] Terminating pod {pod_id} to ensure zero orphaned costs...")
            terminated = self.engine.terminate_runpod_pod(pod_id)
            self.assertTrue(terminated, f"Failed to terminate pod {pod_id}")
            self.__class__.pod_id_to_cleanup = None
            print(f"[RunPod E2E Teardown] Pod {pod_id} confirmed terminated successfully.\n")


if __name__ == "__main__":
    unittest.main()
