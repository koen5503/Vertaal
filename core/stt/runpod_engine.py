"""RunPod remote GPU WebSocket streaming STT engine with auto-discovery and deployment."""

import asyncio
import json
import os
import time
import urllib.error
import urllib.request
from typing import Optional

try:
    import numpy as np
except ImportError:
    np = None

try:
    import websockets
except ImportError:
    websockets = None

from core.audio import FRAME_DURATION_MS, SAMPLE_RATE
from core.stt.base import BaseSTTEngine
from core.utils import get_runpod_gpu_fallbacks


class RunPodSTTEngine(BaseSTTEngine):
    """RunPod remote streaming engine with auto-discovery and deployment."""

    def __init__(self, engine):
        self.engine = engine
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0

    def discover_runpod_url(self) -> Optional[str]:
        """Auto-discover running RunPod endpoint via GraphQL API."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            return None

        api_key = api_key.strip().strip('\'" ')
        masked_key = api_key[:4] + "***" + api_key[-4:] if len(api_key) > 8 else "INVALID_LENGTH"
        print(f"[RunPod Auto-Discovery] Booting with API Key {masked_key} (len: {len(api_key)})")

        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        query = '''
        query Pods {
          myself {
            pods {
              id
              name
              desiredStatus
              runtime {
                ports {
                  ip
                  isIpPublic
                  privatePort
                  publicPort
                  type
                }
              }
            }
          }
        }
        '''

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        data = json.dumps({"query": query}).encode('utf-8')

        try:
            req = urllib.request.Request(url, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=10) as response:
                result = json.loads(response.read())

            pods = result.get('data', {}).get('myself', {}).get('pods', [])
            print(f"[RunPod Auto-Discovery] Found {len(pods)} pods in your account.")

            for pod in pods:
                pod_id = pod.get('id')
                pod_name = pod.get('name')
                desired = pod.get('desiredStatus')
                print(f"  - Checking Pod: {pod_name} (ID: {pod_id}) | desiredStatus: {desired}")

                if desired != 'RUNNING':
                    continue

                runtime = pod.get('runtime')
                if not runtime:
                    continue

                ports = runtime.get('ports', [])
                for port_map in ports:
                    if port_map.get('privatePort') == 8000:
                        ip = port_map.get('ip')
                        public_port = port_map.get('publicPort')

                        if ip and port_map.get('isIpPublic') and public_port:
                            link = f"ws://{ip}:{public_port}/stream"
                            print(f"[RunPod Auto-Discovery] Found TCP Link: {link}")
                            return link
                        else:
                            link = f"wss://{pod_id}-8000.proxy.runpod.net/stream"
                            print(f"[RunPod Auto-Discovery] Found WSS HTTP Proxy Link: {link}")
                            return link

            print("[RunPod Auto-Discovery] No running pod with port 8000 found.")
        except Exception as e:
            print(f"[RunPod Auto-Discovery] Error fetching API: {e}")

        return None

    async def deploy_runpod_pod(self) -> bool:
        """Deploy a RunPod pod using REST API v2 (for Global Volumes) or legacy GraphQL mutation."""
        api_key = os.getenv("RUNPOD_API_KEY")
        template_id = os.getenv("RUNPOD_TEMPLATE_ID")
        volume_id = os.getenv("RUNPOD_VOLUME_ID")
        dc_id = os.getenv("RUNPOD_DATA_CENTER_ID", "").strip()
        whisper_model = os.getenv("RUNPOD_WHISPER_MODEL", "large-v3")

        preferred_gpu = os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")
        gpu_fallbacks = get_runpod_gpu_fallbacks(preferred_gpu)

        if not api_key or not template_id:
            print("[RunPod Deploy] Error: Missing API_KEY or TEMPLATE_ID in .env")
            return False

        region_display = dc_id if dc_id else "Any Region"
        is_global_volume = bool(volume_id and (volume_id.startswith("cmu") or "global" in volume_id.lower()))

        for gpu_id in gpu_fallbacks:
            if is_global_volume:
                print(f"[RunPod Deploy v2] Attempting to rent {gpu_id} with Global Volume {volume_id}...")
                v2_url = "https://api.runpod.io/v2/pods"
                v2_headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                payload = {
                    "name": "ondertitels-gpu",
                    "templateId": template_id,
                    "gpu": {"id": gpu_id, "count": 1},
                    "mounts": {
                        "global": [{"volumeId": volume_id, "path": "/workspace"}]
                    },
                    "entrypoint": ["/bin/bash", "-c"],
                    "cmd": ["rm -rf /workspace/huggingface/hub/models--Systran--faster-whisper* && exec /app/start.sh"]
                }
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(v2_url, data=data, headers=v2_headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=15) as f:
                        res = json.loads(f.read().decode('utf-8'))
                        pod_id = res.get("id") or (res.get("data", {}).get("id") if isinstance(res.get("data"), dict) else None)
                        print(f"[RunPod Deploy v2] Success! Rented {gpu_id}. New Pod ID: {pod_id}")
                        return True
                except urllib.error.HTTPError as he:
                    err_body = he.read().decode('utf-8', errors='ignore')
                    print(f"[RunPod Deploy v2] HTTP {he.code} for {gpu_id}: {err_body}")
                    if "stock" in err_body.lower() or "supply" in err_body.lower() or "unavailable" in err_body.lower() or he.code == 400:
                        continue
                    return False
                except Exception as e:
                    print(f"[RunPod Deploy v2] System Error trying {gpu_id}: {e}")
                    continue
            else:
                # Fallback to legacy GraphQL mutation
                print(f"[RunPod Deploy GraphQL] Attempting to rent {gpu_id} in {region_display}...")
                url = f"https://api.runpod.io/graphql?api_key={api_key}"
                headers = {
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                }
                vol_clause = f'networkVolumeId: "{volume_id}",' if volume_id else ''
                dc_clause = f'dataCenterId: "{dc_id}",' if dc_id else ''

                mutation = f'''
                mutation {{
                  podFindAndDeployOnDemand(input: {{
                    gpuTypeId: "{gpu_id}",
                    templateId: "{template_id}",
                    {vol_clause}
                    {dc_clause}
                    gpuCount: 1,
                    env: [
                        {{ key: "WHISPER_MODEL", value: "{whisper_model}" }}
                    ]
                  }}) {{
                    id
                    desiredStatus
                  }}
                }}
                '''
                data = json.dumps({"query": mutation}).encode('utf-8')
                req = urllib.request.Request(url, data=data, headers=headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=15) as f:
                        res = json.loads(f.read().decode('utf-8'))
                        if "errors" in res:
                            err_msg = res["errors"][0].get("message", "")
                            if "SUPPLY_CONSTRAINT" in str(res) or "no longer any instances available" in err_msg:
                                print(f"[RunPod Deploy GraphQL] {gpu_id} out of stock, trying next...")
                                continue
                            else:
                                print(f"[RunPod Deploy GraphQL] API Error for {gpu_id}: {res['errors']}")
                                return False

                        pod_id = res.get("data", {}).get("podFindAndDeployOnDemand", {}).get("id")
                        print(f"[RunPod Deploy GraphQL] Success! Rented {gpu_id}. New Pod ID: {pod_id}")
                        return True
                except Exception as e:
                    print(f"[RunPod Deploy GraphQL] System Error trying {gpu_id}: {e}")
                    continue

        print("[RunPod Deploy] All attempted GPU types are currently unavailable.")
        return False

    async def manage_runpod_deployment(self) -> bool:
        """Attempt to resume paused pods, terminates them if unavailable, and deploys a new one if needed."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            return False
        api_key = api_key.strip().strip('\'" ')

        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # 1. Get all EXITED pods
        query = '{ myself { pods { id name desiredStatus } } }'
        req = urllib.request.Request(url, data=json.dumps({"query": query}).encode('utf-8'), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pods = json.loads(response.read()).get('data', {}).get('myself', {}).get('pods', [])
        except Exception as e:
            print(f"[RunPod Deploy] Error fetching pods for resume check: {e}")
            pods = []

        exited_pods = [p for p in pods if p.get('desiredStatus') == 'EXITED']

        for pod in exited_pods:
            pod_id = pod['id']
            print(f"[RunPod Deploy] Attempting to resume paused pod {pod_id}...")

            resume_query = f'mutation {{ podResume(input: {{ podId: "{pod_id}", gpuCount: 1 }}) {{ id desiredStatus }} }}'
            req_resume = urllib.request.Request(url, data=json.dumps({"query": resume_query}).encode('utf-8'), headers=headers)
            try:
                with urllib.request.urlopen(req_resume, timeout=15) as f:
                    res = json.loads(f.read().decode('utf-8'))
                    if "errors" not in res:
                        print(f"[RunPod Deploy] Successfully resumed pod {pod_id}!")
                        return True
                    else:
                        err_msg = res["errors"][0].get("message", "")
                        print(f"[RunPod Deploy] Notice: Could not resume {pod_id} (Reason: {err_msg}). Leaving pod intact.")
            except Exception as e:
                print(f"[RunPod Deploy] System error while managing pod {pod_id}: {e}")
                continue

        print("[RunPod Deploy] No paused pods could be resumed. Falling back to renting a fresh pod...")
        return await self.deploy_runpod_pod()

    async def run(self) -> None:
        """Stream VAD-segmented audio to remote NVIDIA RunPod engine."""
        wss_url = os.getenv("RUNPOD_WSS_URL")
        if not wss_url:
            wss_url = self.discover_runpod_url()

        # Auto-deploy if enabled and no pod found
        auto_deploy_val = os.getenv("AUTO_DEPLOY_RUNPOD", "false").lower()
        if not wss_url and auto_deploy_val == "true":
            print("[RunPod] No active pod found. Attempting Auto-Deployment/Resume...")
            if await self.manage_runpod_deployment():
                print("[RunPod] Deployment successful. Waiting for pod to initialize (this can take 1-3 minutes)...")
                for i in range(30):
                    await asyncio.sleep(10)
                    wss_url = self.discover_runpod_url()
                    if wss_url:
                        print("[RunPod] Pod network is ready. Waiting 20s extra for Whisper models to load...")
                        await asyncio.sleep(20)
                        print(f"[RunPod] Ready to connect after {(i + 1) * 10 + 20}s!")
                        break
                    print(f"  ...still waiting for network ({(i + 1) * 10}s)...")

                if not wss_url:
                    print("[RunPod] Timeout: Pod did not become ready in time.")
            else:
                print("[RunPod] Auto-Deployment failed.")

        if not wss_url:
            wss_url = "ws://127.0.0.1:8888/stream"
            print("No RUNPOD_API_KEY or RUNPOD_WSS_URL found. Defaulting to local.")

        loop = asyncio.get_event_loop()

        while True:
            try:
                print(f"Connecting to RunPod Server: {wss_url}")
                async with websockets.connect(wss_url) as ws:
                    print("Connected to RunPod.")

                    self.engine.session_start_time = time.time()
                    await ws.send(json.dumps({
                        "action": "config",
                        "reset_timer": True,
                        "source_lang": self.engine.source_lang,
                        "target_lang_1": self.engine.target_lang_1,
                        "target_lang_2": self.engine.target_lang_2,
                        "target_lang_3": self.engine.target_lang_3,
                        "target_lang_4": self.engine.target_lang_4,
                        "enable_trans_3": getattr(self.engine, 'enable_trans_3', True),
                        "enable_trans_4": getattr(self.engine, 'enable_trans_4', True)
                    }))
                    print("[RunPod] Connected & Session timer reset (full 2-hour window started).")

                    async def ws_receiver():
                        try:
                            async for msg in ws:
                                data = json.loads(msg)
                                await self.engine.broadcast(data)
                        except Exception as e:
                            print(f"RunPod WS Recv Error: {e}")

                    recv_task = asyncio.create_task(ws_receiver())

                    chunk_count = 0
                    self.audio_buffer = bytearray()
                    self.silence_frames = 0
                    self.speech_frames = 0

                    while not self.engine.restart_required:
                        if not self.engine.audio_stream.running:
                            await asyncio.sleep(0.5)
                            continue

                        if self.engine.is_paused:
                            await asyncio.sleep(0.1)
                            continue

                        if time.time() - self.engine.session_start_time > self.engine.MAX_DURATION_SECONDS:
                            self.engine.is_paused = True
                            await self.engine.broadcast({"action": "auto_paused", "reason": f"{int(self.engine.MAX_DURATION_SECONDS / 60)}_min_limit"})
                            continue

                        chunk = await loop.run_in_executor(None, self.engine.audio_stream.read_chunk)
                        if not chunk:
                            continue

                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            rms = self.engine.audio_stream.calculate_rms(chunk)
                            await self.engine.broadcast({"type": "volume", "rms": rms})

                        is_speech = self.engine.audio_stream.is_speech(chunk)
                        if is_speech:
                            self.silence_frames = 0
                            self.speech_frames += 1
                        else:
                            self.silence_frames += 1

                        self.audio_buffer.extend(chunk)

                        # Keep max 1.0s pre-roll silence while waiting for speech
                        if self.speech_frames == 0:
                            PRE_ROLL_BYTES = 32000
                            if len(self.audio_buffer) > PRE_ROLL_BYTES:
                                self.audio_buffer = self.audio_buffer[-PRE_ROLL_BYTES:]

                        silence_ms = self.silence_frames * FRAME_DURATION_MS
                        buffer_duration_s = len(self.audio_buffer) / (SAMPLE_RATE * 2)

                        hard_cap = buffer_duration_s >= self.engine.max_local_chunk_s
                        force_flush = silence_ms >= self.engine.max_silence_flush_ms

                        if (hard_cap or force_flush or silence_ms >= self.engine.silence_threshold_ms) and self.speech_frames > 0:
                            if force_flush and self.speech_frames < 3:
                                self.audio_buffer = bytearray()
                                self.speech_frames = 0
                                self.silence_frames = 0
                                continue

                            met_speech_req = self.speech_frames >= self.engine.min_speech_frames or ((force_flush or hard_cap) and self.speech_frames >= 3)
                            if not met_speech_req:
                                continue

                            if buffer_duration_s < self.engine.min_local_chunk_s and not force_flush and not hard_cap:
                                continue

                            print(f"[RunPod] Sending {buffer_duration_s:.1f}s of audio with {self.speech_frames} speech frames...")
                            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0

                            await ws.send(audio_np.tobytes())

                            self.audio_buffer = bytearray()
                            self.silence_frames = 0
                            self.speech_frames = 0

                    recv_task.cancel()
                    self.engine.restart_required = False
                    if not self.engine.audio_stream.running:
                        self.engine.audio_stream.start()

            except Exception as e:
                print(f"RunPod Server Connection Error. Retrying in 3s... ({e})")
                discovered = self.discover_runpod_url()
                if discovered and discovered != wss_url:
                    print(f"[RunPod] Discovered direct accessible endpoint: {discovered}")
                    wss_url = discovered
                await asyncio.sleep(3)
