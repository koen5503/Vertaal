"""RunPod remote GPU WebSocket streaming STT engine with auto-discovery and deployment."""

import asyncio
import json
import os
import string
import time
import urllib.error
import urllib.request
from typing import Dict, Optional, Set

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
from core.stt.local_engine import filter_anti_hallucination
from core.utils import get_runpod_gpu_fallbacks


def restitch_segment_text(prev_text: str, incoming_text: str) -> str:
    """Restitch incoming transcript or translation against previous text without duplication."""
    if not prev_text:
        return incoming_text.strip()
    if not incoming_text:
        return prev_text.strip()

    prev_clean = prev_text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    inc_clean = incoming_text.translate(str.maketrans('', '', string.punctuation)).lower().strip()

    prev_words = prev_clean.split()
    inc_words = inc_clean.split()
    orig_inc_words = incoming_text.strip().split()

    if not prev_words:
        return incoming_text.strip()
    if not inc_words:
        return prev_text.strip()

    # 1. Check if incoming_text is a full revised transcript of the buffer
    # A. Direct string containment or prefix match
    if inc_clean.startswith(prev_clean[:20]) or prev_clean in inc_clean:
        return incoming_text.strip()

    # B. Word-level prefix overlap (first 5 words >= 60% match)
    check_len = min(5, len(prev_words), len(inc_words))
    if check_len >= 2:
        matching_head = sum(1 for i in range(check_len) if prev_words[i] == inc_words[i])
        if matching_head / check_len >= 0.6:
            return incoming_text.strip()

    # C. Global word containment (if incoming has >= len(prev_words) and >= 50% unique words)
    if len(inc_words) >= len(prev_words):
        prev_set = set(prev_words)
        inc_set = set(inc_words)
        if len(prev_set.intersection(inc_set)) / len(prev_set) >= 0.5:
            return incoming_text.strip()

    # 2. Incoming is a tail delta: find exact word overlap between end of prev_words and start of inc_words
    max_overlap = min(len(prev_words), len(inc_words))
    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == inc_words[:i]:
            tail = " ".join(orig_inc_words[i:])
            return f"{prev_text} {tail}".strip() if tail else prev_text

    # 3. Fuzzy tail overlap (tail sub-sequence appearing near start of incoming)
    for k in range(min(5, len(prev_words)), 1, -1):
        tail_sub = prev_words[-k:]
        search_limit = min(len(inc_words) - k + 1, 5)
        for start_idx in range(search_limit):
            if inc_words[start_idx:start_idx + k] == tail_sub:
                tail = " ".join(orig_inc_words[start_idx + k:])
                return f"{prev_text} {tail}".strip() if tail else prev_text

    # 4. Independent appended clause
    return f"{prev_text} {incoming_text.strip()}".strip()


class RunPodSTTEngine(BaseSTTEngine):
    """RunPod remote streaming engine with auto-discovery and deployment."""

    def __init__(self, engine):
        self.engine = engine
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = 10
        self.active_seg_id: Optional[str] = None
        self.pending_seg_id: Optional[str] = None
        self.pending_status: str = "final"
        self.server_to_client_id: Dict[str, str] = {}
        self.segment_full_transcripts: Dict[str, str] = {}
        self.segment_translations: Dict[str, Dict[str, str]] = {}
        self.segment_status: Dict[str, str] = {}
        self.suppressed_seg_ids: Set[str] = set()
        self.finalized_segments: Set[str] = set()
        self.last_transcript: str = ""
        self.last_interim_s: float = 0.0
        self.is_current_interim: bool = False

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
                    if port_map.get('privatePort') == 8000 and port_map.get('type') != 'udp':
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
        wss_url = self.discover_runpod_url()
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
            wss_url = os.getenv("RUNPOD_WSS_URL")

        if not wss_url:
            wss_url = "ws://127.0.0.1:8888/stream"
            print("No RUNPOD_API_KEY or RUNPOD_WSS_URL found. Defaulting to local.")

        loop = asyncio.get_event_loop()

        while True:
            try:
                print(f"Connecting to RunPod Server: {wss_url}")
                async with websockets.connect(wss_url, ping_interval=None) as ws:
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
                                server_id = data.get("id")
                                if server_id:
                                    if server_id not in self.server_to_client_id:
                                        self.server_to_client_id[server_id] = self.pending_seg_id or server_id
                                        if len(self.server_to_client_id) > 200:
                                            first_key = next(iter(self.server_to_client_id))
                                            del self.server_to_client_id[first_key]
                                    data["id"] = self.server_to_client_id[server_id]

                                client_seg_id = data.get("id")

                                if client_seg_id and client_seg_id in self.suppressed_seg_ids:
                                    print(f"[RunPod STT] Dropping message for suppressed segment '{client_seg_id}'")
                                    continue

                                if client_seg_id:
                                    if client_seg_id in self.finalized_segments:
                                        data["status"] = "final"
                                        self.segment_status[client_seg_id] = "final"
                                    elif "col1_text" in data:
                                        if self.pending_status:
                                            data["status"] = self.pending_status
                                        self.segment_status[client_seg_id] = data.get("status", "final")
                                    else:
                                        # Translation message inherits the segment's actual status (interim or final)
                                        data["status"] = self.segment_status.get(client_seg_id, self.pending_status or "final")

                                is_final = data.get("status") == "final"

                                # Re-stitch transcripts when RunPod strips overlap on subsequent chunks for the same segment
                                if "col1_text" in data and client_seg_id:
                                    incoming_text = data["col1_text"].strip()
                                    prev_text = self.segment_full_transcripts.get(client_seg_id, "")
                                    full_text = restitch_segment_text(prev_text, incoming_text)

                                    # For ongoing segments, last_transcript prompt leakage was already handled on chunk 1
                                    effective_last_transcript = "" if prev_text else self.last_transcript
                                    filtered_text = filter_anti_hallucination(
                                        full_text,
                                        last_transcript=effective_last_transcript,
                                        glossary_phrases=getattr(self.engine, "glossary_phrases", None),
                                    )

                                    if not filtered_text:
                                        print(f"[RunPod STT] Suppressed hallucination segment '{client_seg_id}': '{full_text}'")
                                        self.suppressed_seg_ids.add(client_seg_id)
                                        if len(self.suppressed_seg_ids) > 200:
                                            oldest_suppressed = next(iter(self.suppressed_seg_ids))
                                            self.suppressed_seg_ids.discard(oldest_suppressed)
                                        continue

                                    if is_final:
                                        self.last_transcript = filtered_text[-200:]
                                        if hasattr(self.engine, "last_transcript"):
                                            self.engine.last_transcript = self.last_transcript

                                    self.segment_full_transcripts[client_seg_id] = filtered_text
                                    data["col1_text"] = filtered_text

                                # Re-stitch translation columns if trimmed on subsequent chunks
                                if client_seg_id:
                                    for col_key in ("col2_text", "col3_text", "col4_text", "col5_text"):
                                        if col_key in data and data[col_key]:
                                            incoming_col = str(data[col_key]).strip()
                                            prev_col = self.segment_translations.setdefault(client_seg_id, {}).get(col_key, "")
                                            full_col = restitch_segment_text(prev_col, incoming_col)
                                            self.segment_translations[client_seg_id][col_key] = full_col
                                            data[col_key] = full_col

                                # Log STT transcripts, translation columns, and generic payloads
                                if "col1_text" in data:
                                    status_tag = data.get("status", "final")
                                    print(f"[RunPod STT] [{status_tag}] ({data.get('id')}): '{data['col1_text']}'")
                                elif any(k in data for k in ("col2_text", "col3_text", "col4_text", "col5_text")):
                                    for k in ("col2_text", "col3_text", "col4_text", "col5_text"):
                                        if k in data:
                                            print(f"[RunPod Trans] ({data.get('id')}, {k}): '{data[k]}'")
                                elif data.get("action") == "update_translation":
                                    print(f"[RunPod Trans] ({data.get('id')}, {data.get('col_key')}): '{data.get('text')}'")
                                else:
                                    print(f"[RunPod Msg] Keys: {list(data.keys())} -> {data}")

                                await self.engine.broadcast(data)

                                # Bounded FIFO cleanup for segment text state
                                if len(self.segment_full_transcripts) > 100:
                                    oldest = next(iter(self.segment_full_transcripts))
                                    del self.segment_full_transcripts[oldest]

                                if len(self.segment_translations) > 100:
                                    oldest = next(iter(self.segment_translations))
                                    del self.segment_translations[oldest]

                                if len(self.segment_status) > 100:
                                    oldest = next(iter(self.segment_status))
                                    del self.segment_status[oldest]

                                if len(self.suppressed_seg_ids) > 100:
                                    oldest = next(iter(self.suppressed_seg_ids))
                                    self.suppressed_seg_ids.discard(oldest)

                                if len(self.finalized_segments) > 100:
                                    oldest = next(iter(self.finalized_segments))
                                    self.finalized_segments.discard(oldest)
                        except Exception as e:
                            print(f"RunPod WS Recv Error: {e}")

                    recv_task = asyncio.create_task(ws_receiver())

                    chunk_count = 0
                    self.audio_buffer = bytearray()
                    self.silence_frames = 0
                    self.speech_frames = 0
                    self.active_seg_id = None
                    self.pending_seg_id = None
                    self.pending_status = "final"
                    self.server_to_client_id = {}
                    self.segment_full_transcripts = {}
                    self.segment_translations = {}
                    self.segment_status = {}
                    self.suppressed_seg_ids = set()
                    self.finalized_segments = set()
                    self.last_transcript = ""
                    self.last_interim_s = 0.0
                    self.is_current_interim = False

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

                        is_natural_pause = (silence_ms >= self.engine.silence_threshold_ms and buffer_duration_s >= self.engine.min_local_chunk_s)
                        is_force_flush = (silence_ms >= self.engine.max_silence_flush_ms)
                        is_hard_ceiling = (buffer_duration_s >= self.engine.max_local_chunk_s)
                        is_interim_interval = (
                            (buffer_duration_s - self.last_interim_s >= self.engine.interim_chunk_s)
                            and not is_natural_pause
                            and not is_hard_ceiling
                        )

                        # --- Finalization Trigger (Natural Pause, Max Silence Flush, or Hard Ceiling) ---
                        if (is_natural_pause or is_force_flush or is_hard_ceiling) and self.speech_frames > 0:
                            if is_force_flush and self.speech_frames < 3:
                                self.audio_buffer = bytearray()
                                self.speech_frames = 0
                                self.silence_frames = 0
                                self.active_seg_id = None
                                self.last_interim_s = 0.0
                                self.is_current_interim = False
                                continue

                            met_speech_req = self.speech_frames >= getattr(self.engine, 'min_speech_frames', self.min_speech_frames) or ((is_force_flush or is_hard_ceiling) and self.speech_frames >= 3)
                            if not met_speech_req:
                                continue

                            if not self.active_seg_id:
                                self.active_seg_id = f"seg_{int(time.time()*1000)}"
                            seg_id = self.active_seg_id
                            self.finalized_segments.add(seg_id)
                            if len(self.finalized_segments) > 200:
                                oldest = next(iter(self.finalized_segments))
                                self.finalized_segments.discard(oldest)
                            self.pending_seg_id = seg_id
                            self.pending_status = "final"
                            self.is_current_interim = False
                            reason = "hard_ceiling" if is_hard_ceiling else ("force_flush" if is_force_flush else f"pause_{silence_ms}ms")
                            print(f"[RunPod] Finalizing {seg_id} ({reason}, dur={buffer_duration_s:.1f}s, speech={self.speech_frames}f)...")

                            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            await ws.send(audio_np.tobytes())

                            self.audio_buffer = bytearray()
                            self.silence_frames = 0
                            self.speech_frames = 0
                            self.active_seg_id = None
                            self.last_interim_s = 0.0

                        # --- Incremental Interim Trigger (Option B: continuous speech reaching interim interval) ---
                        elif is_interim_interval and self.speech_frames >= getattr(self.engine, 'min_speech_frames', self.min_speech_frames):
                            if not self.active_seg_id:
                                self.active_seg_id = f"seg_{int(time.time()*1000)}"
                            self.pending_seg_id = self.active_seg_id
                            self.pending_status = "interim"
                            self.is_current_interim = True

                            print(f"[RunPod] Interim update for {self.active_seg_id} (dur={buffer_duration_s:.1f}s, speech={self.speech_frames}f)...")
                            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            await ws.send(audio_np.tobytes())

                            self.last_interim_s = buffer_duration_s

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
