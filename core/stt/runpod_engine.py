"""RunPod remote GPU WebSocket streaming STT engine with auto-discovery and deployment."""

import asyncio
from collections import deque
import difflib
import json
import os
import re
import string
import time
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Set, Tuple

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
from core.stt.local_engine import filter_anti_hallucination, TAIL_HALLUCINATION_PATTERN
from core.translation import clean_translation_meta
from core.utils import get_runpod_gpu_fallbacks


def condense_runpod_logs(raw_logs: str) -> Tuple[str, List[str]]:
    """Clean ANSI codes and raw cursor moves from RunPod container logs, detect boot stage, and return last lines."""
    if not raw_logs or not isinstance(raw_logs, str):
        return "Pod initialiseren...", []

    clean_lines: List[str] = []
    ansi_regex = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')

    raw_split = re.split(r'[\r\n]+', raw_logs)
    last_line = None
    for raw_line in raw_split:
        line = ansi_regex.sub('', raw_line).strip()
        if not line:
            continue
        # Ignore pure progress-bar lines and curl download noise
        if re.search(r'^\d+%.*\[.*it/s\]', line) or re.search(r'^[█▉▊▋▌▍▎▏\s\|]+$', line):
            continue
        if re.search(r'^[#O\s\-\=\.0-9%]+$', line) and ('#' in line or '%' in line):
            continue
        if re.search(r'^\s*#+\s*\d+\.?\d*%', line):
            continue
        if line == last_line:
            continue
        last_line = line
        clean_lines.append(line)

    full_text = " ".join(clean_lines).lower()

    # Look for the most recent step header, e.g. "=== [5/8] Downloading Faster-Whisper large-v3 on local NVMe SSD ==="
    latest_step = None
    step_pattern = re.compile(r'===+\s*(\[\d+/\d+\].*?)\s*===+', re.IGNORECASE)
    for line in clean_lines:
        m = step_pattern.search(line)
        if m:
            latest_step = m.group(1).strip()

    # Check for active sub-steps in recent lines to append to stage badge
    progress_info = None
    for line in reversed(clean_lines[-6:]):
        l_lower = line.lower()
        if "verifying sha256" in l_lower:
            progress_info = "Verifiëren van checksums (SHA256)"
            break
        elif "pulling manifest" in l_lower or "writing manifest" in l_lower:
            progress_info = "Ollama manifest bijwerken"
            break
        elif "staging faster-whisper" in l_lower:
            progress_info = "Faster-Whisper kopiëren naar NVMe SSD"
            break
        elif "staging ollama" in l_lower:
            progress_info = "Ollama models kopiëren naar NVMe SSD"
            break
        elif "restoring ollama engine" in l_lower:
            progress_info = "Ollama binaries herstellen uit cache"
            break
        elif "downloading ollama" in l_lower or "downloading ollama-linux" in l_lower:
            progress_info = "Ollama downloaden van web"
            break
        elif "installing ollama engine" in l_lower:
            progress_info = "Ollama engine installeren"
            break
        elif "extracting python virtual environment" in l_lower:
            progress_info = "Python venv uitpakken naar NVMe SSD"
            break
        elif "syncing application code" in l_lower:
            progress_info = "Applicatiecode synchroniseren"
            break
        elif "starting services" in l_lower or "ollama serve" in l_lower:
            progress_info = "Services opstarten"
            break
        elif "loading faster-whisper" in l_lower:
            progress_info = "Faster-Whisper in GPU laden"
            break
        elif ("pulling" in l_lower or "downloading" in l_lower) and not line.startswith("==="):
            progress_info = line[:55].strip()
            break

    # Fatal error detection across recent log lines
    error_line = None
    for line in reversed(clean_lines[-30:]):
        l_lower = line.lower()
        if (
            "error:" in l_lower
            or "traceback (most recent call last)" in l_lower
            or "modulenotfounderror:" in l_lower
            or "permission denied:" in l_lower
            or "fatal error" in l_lower
            or "fatal:" in l_lower
            or "command not found" in l_lower
            or "killed" in l_lower.split()
            or (("failed" in l_lower or "failure" in l_lower) and not any(ok in l_lower for ok in ["0 failed", "failed: 0", "retry", "fallback"]))
        ):
            error_line = line.strip()
            break

    if error_line:
        stage = f"Fout in container: {error_line[:120]}"
        return stage, clean_lines[-6:]

    if any(k in full_text for k in ["fastapi", "application startup complete", "listening on", "uvicorn running", "uvicorn", "launching live translation server"]):
        stage = "Server operationeel op poort 8000"
    elif latest_step:
        stage = f"Initialisatie {latest_step}"
        if progress_info and progress_info.lower() not in latest_step.lower():
            stage += f" ({progress_info})"
    elif any(k in full_text for k in ["automated setup_once", "first run detected"]):
        stage = "Automatische eerste initialisatie van Global Volume (duurt ~3-5m)..."
    elif any(k in full_text for k in ["faster-whisper", "whisper model loaded", "loading faster-whisper", "aya-expanse", "model list cache"]):
        stage = "AI-modellen laden in GPU VRAM (large-v3 / aya-expanse)"
    elif any(k in full_text for k in ["start container", "create container", "starting container", "created container"]):
        stage = "Container starten op host"
    elif any(k in full_text for k in ["mounting at /workspace", "global store"]):
        stage = "Global storage koppelen aan /workspace"
    elif any(k in full_text for k in ["pulling", "download", "extracting", "docker", "layers"]):
        stage = "Docker container layers downloaden"
    elif any(k in full_text for k in ["discovering available gpus", "nvidia-smi", "nvidia"]):
        stage = "GPU initialisatie"
    else:
        stage = "Pod initialiseren..."

    return stage, clean_lines[-6:]


def is_word_stem_match(w1: str, w2: str) -> bool:
    """Return True if two words share a common root or one is an incomplete/truncated form of the other."""
    if w1 == w2:
        return True
    min_len = min(len(w1), len(w2))
    if min_len < 3:
        return False
    # If one starts with the other (e.g. bepalen / bepalend, geloof / gelovige, etc.)
    if w1.startswith(w2) or w2.startswith(w1):
        return True
    # Shared prefix of >= 2 chars and high similarity ratio (e.g. paas / pasen, lez / lezen)
    if w1[:2] == w2[:2]:
        ratio = difflib.SequenceMatcher(None, w1, w2).ratio()
        if ratio >= 0.65:
            return True
    return False


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
    orig_prev_words = prev_text.strip().split()
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

    # 2. Exact word overlap between end of prev_words and start of inc_words
    max_overlap = min(len(prev_words), len(inc_words))
    for i in range(max_overlap, 0, -1):
        if prev_words[-i:] == inc_words[:i]:
            tail = " ".join(orig_inc_words[i:])
            return f"{prev_text} {tail}".strip() if tail else prev_text

    # 3. Stem / truncated boundary word overlap (e.g. "... Alles bepalen." + "bepalend." -> "... Alles bepalend.")
    for i in range(min(5, len(prev_words), len(inc_words)), 0, -1):
        earlier_match = (i == 1) or (prev_words[-i:-1] == inc_words[:i-1])
        if earlier_match and is_word_stem_match(prev_words[-1], inc_words[i-1]):
            trimmed_prev = " ".join(orig_prev_words[:-i])
            tail = " ".join(orig_inc_words)
            return f"{trimmed_prev} {tail}".strip() if trimmed_prev else tail

    # 4. Fuzzy tail overlap (tail sub-sequence appearing near start of incoming)
    for k in range(min(5, len(prev_words)), 1, -1):
        tail_sub = prev_words[-k:]
        search_limit = min(len(inc_words) - k + 1, 5)
        for start_idx in range(search_limit):
            if inc_words[start_idx:start_idx + k] == tail_sub:
                tail = " ".join(orig_inc_words[start_idx + k:])
                return f"{prev_text} {tail}".strip() if tail else prev_text

    # 5. Independent appended clause
    return f"{prev_text} {incoming_text.strip()}".strip()


RUNPOD_BOOT_CMD = '[ -f /workspace/boot.sh ] && bash /workspace/boot.sh || (curl -fsSL https://raw.githubusercontent.com/koen5503/Vertaal/feature/local-mac-ai-pipeline/boot.sh -o /tmp/boot.sh && bash /tmp/boot.sh)'
RUNPOD_OVERRIDE_CMD = RUNPOD_BOOT_CMD


class RunPodSTTEngine(BaseSTTEngine):
    """RunPod remote streaming engine with auto-discovery and deployment."""

    def __init__(self, engine=None):
        self.engine = engine
        self.active_pod_id: Optional[str] = None
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.min_speech_frames = 10
        self.active_seg_id: Optional[str] = None
        self.pending_seg_id: Optional[str] = None
        self.pending_status: str = "final"
        self.server_to_client_id: Dict[str, str] = {}
        self.server_chunk_is_delta: Dict[str, bool] = {}
        self.segment_last_chunk_is_delta: Dict[str, bool] = {}
        self.segment_full_transcripts: Dict[str, str] = {}
        self.segment_translations: Dict[str, Dict[str, str]] = {}
        self.segment_status: Dict[str, str] = {}
        self.suppressed_seg_ids: Set[str] = set()
        self.finalized_segments: Set[str] = set()
        self.last_transcript: str = ""
        self.last_interim_s: float = 0.0
        self.is_current_interim: bool = False
        self.silence_heartbeat_sec: float = float(os.getenv("SILENCE_HEARTBEAT_SEC", "60"))
        self.last_speech_or_heartbeat_time: float = time.time()
        self.heartbeat_seg_ids: Set[str] = set()
        self.heartbeat_probe_sent_time: float = 0.0
        self.intentional_shutdown: bool = False
        self.segment_sent_time: Dict[str, float] = {}
        self.segment_stt_time: Dict[str, float] = {}
        self.last_stt_latency_ms: Optional[int] = None
        self.last_trans_latency_ms: Optional[int] = None
        self.last_deploy_error: Optional[str] = None
        self.active_gpu_name: Optional[str] = os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")
        if self.engine and hasattr(self.engine, "audio_stream"):
            self.engine.audio_stream.is_connected = False

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
              machine {
                gpuDisplayName
              }
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

            data_obj = result.get('data') or {}
            myself_obj = data_obj.get('myself') or {}
            pods = myself_obj.get('pods') or []
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

                        # Extract GPU model name from pod machine metadata
                        pod_gpu = pod.get("machine", {}).get("gpuDisplayName") if isinstance(pod.get("machine"), dict) else None
                        self.active_gpu_name = pod_gpu or self.active_gpu_name or os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")

                        if ip and port_map.get('isIpPublic') and public_port:
                            link = f"ws://{ip}:{public_port}/stream"
                            print(f"[RunPod Auto-Discovery] Found TCP Link: {link} (GPU: {self.active_gpu_name})")
                            self.active_pod_id = pod_id
                            return link
                        else:
                            link = f"wss://{pod_id}-8000.proxy.runpod.net/stream"
                            print(f"[RunPod Auto-Discovery] Found WSS HTTP Proxy Link: {link} (GPU: {self.active_gpu_name})")
                            self.active_pod_id = pod_id
                            return link

            print("[RunPod Auto-Discovery] No running pod with port 8000 found.")
        except Exception as e:
            print(f"[RunPod Auto-Discovery] Error fetching API: {e}")

        return None

    async def deploy_runpod_pod(self) -> bool:
        """Deploy a RunPod pod using REST API v2 (for Global Volumes) or legacy GraphQL mutation."""
        self.last_deploy_error = None
        api_key = os.getenv("RUNPOD_API_KEY")
        template_id = os.getenv("RUNPOD_TEMPLATE_ID")
        volume_id = os.getenv("RUNPOD_VOLUME_ID")
        dc_id = os.getenv("RUNPOD_DATA_CENTER_ID", "").strip()
        whisper_model = os.getenv("RUNPOD_WHISPER_MODEL", "large-v3")

        preferred_gpu = os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")
        gpu_fallbacks = get_runpod_gpu_fallbacks(preferred_gpu)

        if not api_key or not template_id:
            self.last_deploy_error = "Missing RUNPOD_API_KEY or RUNPOD_TEMPLATE_ID in environment."
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
                    }
                }
                data = json.dumps(payload).encode('utf-8')
                req = urllib.request.Request(v2_url, data=data, headers=v2_headers, method="POST")

                try:
                    with urllib.request.urlopen(req, timeout=15) as f:
                        res = json.loads(f.read().decode('utf-8'))
                        pod_id = res.get("id") or (res.get("data", {}).get("id") if isinstance(res.get("data"), dict) else None)
                        if pod_id:
                            self.active_pod_id = str(pod_id)
                            self.active_gpu_name = gpu_id
                        print(f"[RunPod Deploy v2] Success! Rented {gpu_id}. New Pod ID: {pod_id}")
                        return True
                except urllib.error.HTTPError as he:
                    err_body = he.read().decode('utf-8', errors='ignore')
                    print(f"[RunPod Deploy v2] HTTP {he.code} for {gpu_id}: {err_body}")
                    self.last_deploy_error = f"GPU {gpu_id}: HTTP {he.code} - {err_body}"
                    if "stock" in err_body.lower() or "supply" in err_body.lower() or "unavailable" in err_body.lower() or he.code == 400:
                        continue
                    return False
                except Exception as e:
                    print(f"[RunPod Deploy v2] System Error trying {gpu_id}: {e}")
                    self.last_deploy_error = f"GPU {gpu_id}: {e}"
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
                            self.last_deploy_error = f"GPU {gpu_id}: GraphQL error - {err_msg or res.get('errors')}"
                            if "SUPPLY_CONSTRAINT" in str(res) or "no longer any instances available" in err_msg:
                                print(f"[RunPod Deploy GraphQL] {gpu_id} out of stock, trying next...")
                                continue
                            else:
                                print(f"[RunPod Deploy GraphQL] API Error for {gpu_id}: {res['errors']}")
                                return False

                        pod_id = res.get("data", {}).get("podFindAndDeployOnDemand", {}).get("id")
                        if pod_id:
                            self.active_pod_id = str(pod_id)
                            self.active_gpu_name = gpu_id
                        print(f"[RunPod Deploy GraphQL] Success! Rented {gpu_id}. New Pod ID: {pod_id}")
                        return True
                except Exception as e:
                    print(f"[RunPod Deploy GraphQL] System Error trying {gpu_id}: {e}")
                    self.last_deploy_error = f"GPU {gpu_id}: {e}"
                    continue

        if not self.last_deploy_error:
            self.last_deploy_error = f"Alle geprobeerde GPUs ({', '.join(gpu_fallbacks)}) zijn momenteel niet beschikbaar op RunPod."
        print(f"[RunPod Deploy] Deployment failed: {self.last_deploy_error}")
        return False

    async def manage_runpod_deployment(self) -> bool:
        """Adopts existing running pod if available, terminates exited pods, and rents a fresh pod."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            return False
        api_key = api_key.strip().strip('\'" ')

        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        # 1. Fetch current pods
        query = '{ myself { pods { id name desiredStatus } } }'
        req = urllib.request.Request(url, data=json.dumps({"query": query}).encode('utf-8'), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pods = json.loads(response.read()).get('data', {}).get('myself', {}).get('pods', [])
        except Exception as e:
            print(f"[RunPod Deploy] Error fetching pods: {e}")
            pods = []

        # 2. Check if an active pod is ALREADY RUNNING
        running_pods = [
            p for p in pods
            if p.get('desiredStatus') == 'RUNNING' and ('ondertitels' in p.get('name', '').lower() or p.get('id') == self.active_pod_id)
        ]
        if running_pods:
            existing_id = running_pods[0]['id']
            print(f"[RunPod Deploy] Found existing active pod {existing_id}. Attaching to it...")
            self.active_pod_id = str(existing_id)
            return True

        # 3. Terminate any EXITED / stopped pods (Community Cloud pods cannot be resumed)
        print("[RunPod Deploy] Cleaning up any stopped/zombie pods...")
        self.cleanup_exited_pods()

        # 4. Rent fresh on-demand pod
        print("[RunPod Deploy] Renting fresh pod...")
        return await self.deploy_runpod_pod()

    def terminate_runpod_pod(self, pod_id: str) -> bool:
        """Terminate a RunPod pod completely via GraphQL mutation."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key or not pod_id:
            return False

        api_key = api_key.strip().strip('\'" ')
        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        mutation = f'''
        mutation {{
          podTerminate(input: {{podId: "{pod_id}"}})
        }}
        '''
        data = json.dumps({"query": mutation}).encode('utf-8')
        req = urllib.request.Request(url, data=data, headers=headers, method="POST")

        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                res = json.loads(response.read().decode('utf-8'))
                if "errors" in res:
                    print(f"[RunPod Terminate] Error terminating pod {pod_id}: {res['errors']}")
                    return False
                print(f"[RunPod Terminate] Successfully terminated pod {pod_id}.")
                return True
        except Exception as e:
            print(f"[RunPod Terminate] Failed to terminate pod {pod_id}: {e}")
            return False

    def cleanup_exited_pods(self) -> int:
        """Query account for EXITED zombie pods and terminate them completely."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key:
            return 0

        api_key = api_key.strip().strip('\'" ')
        url = f"https://api.runpod.io/graphql?api_key={api_key}"
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        query = '{ myself { pods { id name desiredStatus } } }'
        req = urllib.request.Request(url, data=json.dumps({"query": query}).encode('utf-8'), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=10) as response:
                pods = json.loads(response.read().decode('utf-8')).get('data', {}).get('myself', {}).get('pods', [])
        except Exception as e:
            print(f"[RunPod Cleanup] Error querying pods: {e}")
            return 0

        terminated_count = 0
        for pod in pods:
            if pod.get('desiredStatus') == 'EXITED':
                pod_id = pod.get('id')
                pod_name = pod.get('name', '')
                if pod_id and self.terminate_runpod_pod(pod_id):
                    print(f"[RunPod Cleanup] Terminated zombie pod {pod_id} ({pod_name})")
                    terminated_count += 1

        return terminated_count

    async def stop_runpod_pod(self) -> bool:
        """Stop and terminate the currently active RunPod pod via GraphQL mutation to avoid runaway cloud costs and zombie pods."""
        api_key = os.getenv("RUNPOD_API_KEY")
        if not self.active_pod_id or not api_key:
            return False

        self.intentional_shutdown = True
        pod_id = self.active_pod_id
        loop = asyncio.get_event_loop()
        success = await loop.run_in_executor(None, self.terminate_runpod_pod, pod_id)
        self.active_pod_id = None
        return success

    condense_runpod_logs = staticmethod(condense_runpod_logs)

    def fetch_pod_logs(self, pod_id: str, api_key: Optional[str] = None) -> Tuple[Optional[str], Optional[str], str]:
        """Fetch pod status and runtime container & system logs from RunPod REST API v2 streaming endpoint."""
        if not api_key:
            api_key = os.getenv("RUNPOD_API_KEY")
        if not api_key or not pod_id:
            return None, None, ""

        api_key = api_key.strip().strip('\'" ')

        # 1. Fetch Pod Desired Status via lightweight GraphQL query
        desired_status: Optional[str] = None
        try:
            status_query = 'query { myself { pods { id desiredStatus } } }'
            status_data = json.dumps({"query": status_query}).encode('utf-8')
            status_headers = {
                "Content-Type": "application/json",
                "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            }
            status_req = urllib.request.Request(
                f"https://api.runpod.io/graphql?api_key={api_key}",
                data=status_data,
                headers=status_headers,
                method="POST",
            )
            with urllib.request.urlopen(status_req, timeout=4) as status_resp:
                res = json.loads(status_resp.read().decode('utf-8'))
                pods = res.get('data', {}).get('myself', {}).get('pods', [])
                for p in pods:
                    if p.get('id') == pod_id:
                        desired_status = p.get('desiredStatus')
                        break
        except Exception:
            pass

        # 2. Fetch live logs from RunPod REST API v2 streaming endpoint
        logs_url = f"https://api.runpod.io/v2/pods/{pod_id}/logs"
        logs_headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        }
        logs_req = urllib.request.Request(logs_url, headers=logs_headers, method="GET")

        clean_lines = deque(maxlen=200)
        last_msg: Optional[str] = None
        start_time = time.time()

        try:
            with urllib.request.urlopen(logs_req, timeout=3.0) as response:
                for raw in response:
                    if (time.time() - start_time) > 3.0:
                        break
                    line_str = raw.decode("utf-8", errors="replace").strip() if isinstance(raw, (bytes, bytearray)) else str(raw).strip()
                    if not line_str:
                        continue

                    # Filter curl / progress noise
                    if re.search(r'^[#O\s\-\=\.0-9%]+$', line_str) and ('#' in line_str or '%' in line_str):
                        continue

                    # Handle SSE format (lines beginning with 'data:')
                    if line_str.startswith("data:"):
                        payload_str = line_str[5:].strip()
                        if not payload_str:
                            continue
                        try:
                            payload = json.loads(payload_str)
                            source = (payload.get("source") or "system").upper()
                            text = payload.get("line") or payload.get("message") or ""
                            text = text.strip()
                            if text:
                                if re.search(r'^[#O\s\-\=\.0-9%]+$', text) and ('#' in text or '%' in text):
                                    continue
                                # Deduplicate consecutive identical messages
                                if text == last_msg:
                                    continue
                                last_msg = text
                                clean_lines.append(f"[{source}] {text}")
                        except json.JSONDecodeError:
                            if payload_str != last_msg:
                                last_msg = payload_str
                                clean_lines.append(payload_str)
                    else:
                        # Raw log line without SSE 'data:' prefix
                        if line_str != last_msg:
                            last_msg = line_str
                            clean_lines.append(line_str)
        except Exception:
            # Normal for streaming endpoints upon timeout or connection termination
            pass

        clean_logs_text = "\n".join(clean_lines)

        # Derive runtime status from desired_status and log stage
        stage, _ = self.condense_runpod_logs(clean_logs_text)
        if "fout" in stage.lower() or "error" in stage.lower():
            derived_status = "ERROR"
        elif "Server operationeel" in stage:
            derived_status = "CONNECTED"
        elif any(k in stage for k in ["AI-modellen", "Container starten", "storage", "layers", "GPU initialisatie"]):
            derived_status = desired_status or "RUNNING"
        else:
            derived_status = desired_status or "INITIALIZING"

        return derived_status, desired_status, clean_logs_text

    async def broadcast_boot_telemetry(self, pod_id: Optional[str], stage: str, status: Optional[str] = None, logs: Optional[List[str]] = None):
        """Broadcast boot telemetry to connected clients and store snapshot on engine."""
        boot_data = {
            "type": "telemetry_boot",
            "pod_id": pod_id,
            "stage": stage,
            "status": status or "INITIALIZING",
            "logs": logs or [],
            "gpu_name": self.active_gpu_name or os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090"),
        }
        if self.engine:
            if hasattr(self.engine, "last_boot_telemetry"):
                self.engine.last_boot_telemetry = boot_data
            if hasattr(self.engine, "broadcast") and callable(self.engine.broadcast):
                try:
                    res = self.engine.broadcast(boot_data)
                    if asyncio.iscoroutine(res):
                        await res
                except Exception as e:
                    print(f"[RunPod Boot Telemetry] Error broadcasting: {e}")

    async def stop(self) -> None:
        """Stop streaming and terminate remote RunPod pod."""
        await self.stop_runpod_pod()

    async def run(self) -> None:
        """Stream VAD-segmented audio to remote NVIDIA RunPod engine."""
        loop = asyncio.get_event_loop()
        wss_url = self.discover_runpod_url()
        auto_deploy_val = os.getenv("AUTO_DEPLOY_RUNPOD", "false").lower()
        if not wss_url and auto_deploy_val == "true":
            print("[RunPod] No active pod found. Attempting Auto-Deployment/Resume...")
            await self.broadcast_boot_telemetry(self.active_pod_id, "Pod provisioning via REST/GraphQL API...", "STARTING", ["Auto-deployment geactiveerd. Pod aanmaken of opzoeken..."])
            deploy_success = await self.manage_runpod_deployment()
            if deploy_success:
                print("[RunPod] Deployment successful. Waiting for pod to initialize (this can take 1-3 minutes)...")
                start_wait = time.time()
                await self.broadcast_boot_telemetry(self.active_pod_id, "Pod initialiseren... (0s verlopen)", "INITIALIZING", ["Pod actief/aangemaakt. Wachten op netwerk en containers..."])
                for i in range(96):
                    await asyncio.sleep(5)
                    elapsed_sec = int(time.time() - start_wait)
                    if self.active_pod_id:
                        last_status, desired_status, raw_logs = await loop.run_in_executor(None, self.fetch_pod_logs, self.active_pod_id)
                        stage, clean_lines = self.condense_runpod_logs(raw_logs)
                        detailed_stage = f"{stage} ({elapsed_sec}s verlopen)" if "verlopen" not in stage else stage
                        await self.broadcast_boot_telemetry(self.active_pod_id, detailed_stage, last_status or desired_status, clean_lines)
                    wss_url = self.discover_runpod_url()
                    if wss_url:
                        elapsed_sec = int(time.time() - start_wait)
                        print(f"[RunPod] Pod network port 8000 discovered ({elapsed_sec}s). Transitioning to live connection and boot log stream...")
                        break
                    print(f"  ...still waiting for network ({elapsed_sec}s)...")

                if not wss_url:
                    elapsed_sec = int(time.time() - start_wait)
                    print(f"[RunPod] Timeout: Pod did not become ready in time ({elapsed_sec}s).")
                    await self.broadcast_boot_telemetry(self.active_pod_id, f"Time-out: Pod niet tijdig gereed ({elapsed_sec}s)", "TIMEOUT", [f"Time-out bereikt na {elapsed_sec}s."])
            else:
                preferred_gpu = os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090")
                err_detail = getattr(self, "last_deploy_error", None) or f"Kon pod niet provisioneren op RunPod (GPU: {self.active_gpu_name or preferred_gpu})."
                print(f"[RunPod] Auto-Deployment failed: {err_detail}")
                await self.broadcast_boot_telemetry(
                    self.active_pod_id,
                    f"Auto-Deployment mislukt: {self.active_gpu_name or preferred_gpu}",
                    "FAILED",
                    [f"Foutdetails: {err_detail}"]
                )

        if not wss_url:
            wss_url = os.getenv("RUNPOD_WSS_URL")

        if not wss_url:
            wss_url = "ws://127.0.0.1:8888/stream"
            print("No RUNPOD_API_KEY or RUNPOD_WSS_URL found. Defaulting to local.")

        if self.engine and hasattr(self.engine, "audio_stream"):
            self.engine.audio_stream.is_connected = False

        connect_start_time = time.time()
        while True:
            try:
                print(f"Connecting to RunPod Server: {wss_url}")
                async with websockets.connect(
                    wss_url,
                    ping_interval=30,
                    ping_timeout=120,
                    close_timeout=15,
                    max_size=10_000_000,
                ) as ws:
                    print("Connected to RunPod.")
                    await self.broadcast_boot_telemetry(self.active_pod_id, "Server operationeel op poort 8000", "CONNECTED", ["WebSocket verbonden op poort 8000. Pipeline actief."])
                    self.intentional_shutdown = False
                    self.last_speech_or_heartbeat_time = time.time()

                    if self.engine and hasattr(self.engine, "audio_stream"):
                        self.engine.audio_stream.is_connected = True
                        if not self.engine.audio_stream.running:
                            print("[Audio] Starting audio stream / demo file playback now that RunPod is connected.")
                            self.engine.audio_stream.start()

                    self.engine.reset_session_timer(reason="runpod_connected")
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
                    print(f"[RunPod] Connected & Session timer reset ({int(self.engine.initial_session_window_minutes)}m sliding window started).")

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

                                # Silence heartbeat probe interception & telemetry calculation
                                is_probe = (
                                    (client_seg_id and (client_seg_id in self.heartbeat_seg_ids or str(client_seg_id).startswith("probe_")))
                                    or data.get("type") == "probe_response"
                                )
                                if is_probe:
                                    latency_ms = int((time.time() - self.heartbeat_probe_sent_time) * 1000) if self.heartbeat_probe_sent_time else 0
                                    print(f"[Heartbeat] Probe returned successfully (roundtrip: {latency_ms}ms). Pipeline healthy.")
                                    await self.engine.broadcast({
                                        "type": "heartbeat",
                                        "status": "ok",
                                        "latency_ms": latency_ms,
                                        "timestamp": time.time(),
                                    })
                                    if client_seg_id:
                                        self.heartbeat_seg_ids.discard(client_seg_id)
                                    continue

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

                                    # Determine if RunPod server trimmed overlap on this chunk (making STT and its translations delta tails)
                                    is_server_delta = data.get("is_delta", None)
                                    if is_server_delta is not None:
                                        is_delta_chunk = bool(is_server_delta)
                                        if is_delta_chunk and prev_text:
                                            full_text = restitch_segment_text(prev_text, incoming_text)
                                        else:
                                            full_text = incoming_text
                                    else:
                                        full_text = restitch_segment_text(prev_text, incoming_text)
                                        is_delta_chunk = bool(prev_text and incoming_text != full_text)

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

                                    self.engine.extend_session_window(reason="stt_activity")

                                    if server_id:
                                        self.server_chunk_is_delta[server_id] = is_delta_chunk
                                        if len(self.server_chunk_is_delta) > 200:
                                            oldest_chunk = next(iter(self.server_chunk_is_delta))
                                            del self.server_chunk_is_delta[oldest_chunk]
                                    self.segment_last_chunk_is_delta[client_seg_id] = is_delta_chunk
                                    if len(self.segment_last_chunk_is_delta) > 200:
                                        oldest_seg = next(iter(self.segment_last_chunk_is_delta))
                                        del self.segment_last_chunk_is_delta[oldest_seg]

                                    if is_final:
                                        self.last_transcript = filtered_text[-1200:]
                                        if hasattr(self.engine, "last_transcript"):
                                            self.engine.last_transcript = self.last_transcript

                                    self.segment_full_transcripts[client_seg_id] = filtered_text
                                    data["col1_text"] = filtered_text

                                    sent_t = self.segment_sent_time.get(client_seg_id)
                                    stt_latency_ms = int((time.time() - sent_t) * 1000) if sent_t else None
                                    if stt_latency_ms is not None:
                                        self.last_stt_latency_ms = stt_latency_ms
                                    self.segment_stt_time[client_seg_id] = time.time()
                                    if len(self.segment_stt_time) > 200:
                                        oldest_tt = next(iter(self.segment_stt_time))
                                        del self.segment_stt_time[oldest_tt]

                                    await self.engine.broadcast({
                                        "type": "telemetry",
                                        "stt_latency_ms": stt_latency_ms,
                                        "trans_latency_ms": self.last_trans_latency_ms,
                                        "pod_id": self.active_pod_id,
                                        "gpu_name": self.active_gpu_name or os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090"),
                                        "pipeline_mode": getattr(self.engine, "pipeline_mode", "runpod"),
                                    })

                                # Translation columns: stitch if delta tail, replace directly if full revision
                                if client_seg_id:
                                    if data.get("is_delta") is not None:
                                        is_delta = bool(data.get("is_delta"))
                                    else:
                                        is_delta = self.server_chunk_is_delta.get(
                                            server_id,
                                            self.segment_last_chunk_is_delta.get(client_seg_id, False)
                                        ) if server_id else self.segment_last_chunk_is_delta.get(client_seg_id, False)

                                    for col_key in ("col2_text", "col3_text", "col4_text", "col5_text"):
                                        if col_key in data and data[col_key]:
                                            incoming_col = str(data[col_key]).strip()
                                            prev_col = self.segment_translations.setdefault(client_seg_id, {}).get(col_key, "")

                                            if is_delta and prev_col:
                                                # RunPod only translated the delta tail; stitch onto earlier translation
                                                full_col = restitch_segment_text(prev_col, incoming_col)
                                            else:
                                                # RunPod translated the full segment (or chunk 1); direct replacement
                                                full_col = incoming_col

                                            # Strip LLM meta-commentary / conversational lecturing
                                            cleaned_col = clean_translation_meta(full_col)
                                            # Strip trailing broadcast credits / noise artifacts
                                            cleaned_col = TAIL_HALLUCINATION_PATTERN.sub("", cleaned_col).strip()
                                            # Also strip any trailing Gelderland / station / year variations (including Cyrillic)
                                            cleaned_col = re.sub(
                                                r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)(?:tv|тв)?\s*(?:gelderland|gelreland|гельдерланд)(?:\s+\d{4})?[\.\,\!\?\s]*$",
                                                "",
                                                cleaned_col,
                                                flags=re.IGNORECASE
                                            ).strip()

                                            if cleaned_col != full_col:
                                                print(f"[RunPod Trans] Cleaned translation artifact from {col_key}: '{full_col}' -> '{cleaned_col}'")

                                            self.segment_translations.setdefault(client_seg_id, {})[col_key] = cleaned_col
                                            data[col_key] = cleaned_col

                                            if col_key == "col2_text":
                                                stt_t = self.segment_stt_time.get(client_seg_id)
                                                trans_latency_ms = int((time.time() - stt_t) * 1000) if stt_t else None
                                                if trans_latency_ms is not None:
                                                    self.last_trans_latency_ms = trans_latency_ms
                                                await self.engine.broadcast({
                                                    "type": "telemetry",
                                                    "stt_latency_ms": self.last_stt_latency_ms,
                                                    "trans_latency_ms": trans_latency_ms,
                                                    "pod_id": self.active_pod_id,
                                                    "gpu_name": self.active_gpu_name or os.getenv("RUNPOD_GPU_ID", "NVIDIA GeForce RTX 4090"),
                                                    "pipeline_mode": getattr(self.engine, "pipeline_mode", "runpod"),
                                                })

                                if data.get("action") == "update_translation" and "text" in data and data["text"]:
                                    full_t = str(data["text"]).strip()
                                    cleaned_t = clean_translation_meta(full_t)
                                    cleaned_t = TAIL_HALLUCINATION_PATTERN.sub("", cleaned_t).strip()
                                    cleaned_t = re.sub(
                                        r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)(?:tv|тв)?\s*(?:gelderland|gelreland|гельдерланд)(?:\s+\d{4})?[\.\,\!\?\s]*$",
                                        "",
                                        cleaned_t,
                                        flags=re.IGNORECASE
                                    ).strip()
                                    data["text"] = cleaned_t

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
                        except Exception as e:
                            if self.intentional_shutdown:
                                print("[RunPod] WebSocket closed normally after intentional podStop.")
                            else:
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
                    self.server_chunk_is_delta = {}
                    self.segment_last_chunk_is_delta = {}
                    self.segment_full_transcripts = {}
                    self.segment_translations = {}
                    self.segment_status = {}
                    self.suppressed_seg_ids = set()
                    self.finalized_segments = set()
                    self.last_transcript = ""
                    self.last_interim_s = 0.0
                    self.is_current_interim = False
                    self.segment_sent_time = {}
                    self.segment_stt_time = {}
                    self.last_stt_latency_ms = None
                    self.last_trans_latency_ms = None

                    while not self.engine.restart_required:
                        if not self.engine.audio_stream.running:
                            await asyncio.sleep(0.5)
                            continue

                        if self.engine.is_paused:
                            await asyncio.sleep(0.1)
                            continue

                        if time.time() > self.engine.session_expiry_time:
                            print(f"[RunPod] TIMEOUT REACHED (expiry={self.engine.session_expiry_time:.1f}). Auto-pausing & stopping pod.")
                            self.engine.is_paused = True
                            await self.stop_runpod_pod()
                            await self.engine.broadcast({"action": "auto_paused", "reason": f"{int(self.engine.initial_session_window_minutes)}_min_limit"})
                            continue

                        chunk = await loop.run_in_executor(None, self.engine.audio_stream.read_chunk)
                        if not chunk:
                            continue

                        chunk_count += 1
                        if chunk_count % 10 == 0:
                            rms = self.engine.audio_stream.calculate_rms(chunk)
                            await self.engine.broadcast({"type": "volume", "rms": rms})

                        if chunk_count % 30 == 0:
                            silence_sec = round((self.silence_frames * FRAME_DURATION_MS) / 1000.0, 1)
                            next_heartbeat_in = max(0, int(self.silence_heartbeat_sec - (time.time() - self.last_speech_or_heartbeat_time)))
                            await self.engine.broadcast({
                                "type": "telemetry_vad",
                                "silence_sec": silence_sec,
                                "speech_frames": self.speech_frames,
                                "next_heartbeat_in": next_heartbeat_in,
                            })

                        is_speech = self.engine.audio_stream.is_speech(chunk)
                        if is_speech:
                            self.silence_frames = 0
                            self.speech_frames += 1
                            self.last_speech_or_heartbeat_time = time.time()
                        else:
                            self.silence_frames += 1

                        self.audio_buffer.extend(chunk)

                        # Keep max 1.0s pre-roll silence while waiting for speech
                        if self.speech_frames == 0:
                            PRE_ROLL_BYTES = 32000
                            if len(self.audio_buffer) > PRE_ROLL_BYTES:
                                self.audio_buffer = self.audio_buffer[-PRE_ROLL_BYTES:]

                            # Silence heartbeat probe trigger
                            now = time.time()
                            silence_s = now - self.last_speech_or_heartbeat_time
                            if silence_s >= self.silence_heartbeat_sec:
                                probe_id = f"probe_{int(now * 1000)}"
                                self.heartbeat_seg_ids.add(probe_id)
                                self.pending_seg_id = probe_id
                                self.pending_status = "final"
                                self.heartbeat_probe_sent_time = now
                                self.last_speech_or_heartbeat_time = now
                                print(f"[Heartbeat] Sent silence heartbeat probe to RunPod after {silence_s:.1f}s of silence.")

                                if np is not None:
                                    t = np.arange(16000) / 16000.0
                                    probe_np = (0.005 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
                                    probe_bytes = probe_np.tobytes()
                                else:
                                    probe_bytes = b'\x00' * 32000

                                await ws.send(json.dumps({"action": "probe", "id": probe_id}))
                                await ws.send(probe_bytes)

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

                            self.segment_sent_time[seg_id] = time.time()
                            if len(self.segment_sent_time) > 200:
                                oldest_st = next(iter(self.segment_sent_time))
                                del self.segment_sent_time[oldest_st]

                            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            await ws.send(json.dumps({"action": "audio_meta", "seg_id": seg_id, "status": "final"}))
                            await ws.send(audio_np.tobytes())
                            await ws.send(json.dumps({"action": "reset_timer"}))
                            self.engine.extend_session_window(reason="stt_activity")

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

                            self.segment_sent_time[self.active_seg_id] = time.time()
                            if len(self.segment_sent_time) > 200:
                                oldest_st = next(iter(self.segment_sent_time))
                                del self.segment_sent_time[oldest_st]

                            audio_np = np.frombuffer(self.audio_buffer, dtype=np.int16).astype(np.float32) / 32768.0
                            await ws.send(json.dumps({"action": "audio_meta", "seg_id": self.active_seg_id, "status": "interim"}))
                            await ws.send(audio_np.tobytes())
                            await ws.send(json.dumps({"action": "reset_timer"}))
                            self.engine.extend_session_window(reason="stt_activity")

                            self.last_interim_s = buffer_duration_s

                    recv_task.cancel()
                    self.engine.restart_required = False
                    if self.engine and hasattr(self.engine, "audio_stream"):
                        self.engine.audio_stream.is_connected = False
                    if not self.engine.audio_stream.running:
                        self.engine.audio_stream.start()

            except Exception as e:
                if self.engine and hasattr(self.engine, "audio_stream"):
                    self.engine.audio_stream.is_connected = False
                print(f"RunPod Server Connection Error. Retrying in 4s... ({e})")
                discovered = self.discover_runpod_url()
                if discovered and discovered != wss_url:
                    print(f"[RunPod] Discovered direct accessible endpoint: {discovered}")
                    wss_url = discovered

                # Continuously stream live boot telemetry and container logs during connection retries
                if self.active_pod_id:
                    elapsed_sec = int(time.time() - connect_start_time)
                    try:
                        last_status, desired_status, raw_logs = await loop.run_in_executor(None, self.fetch_pod_logs, self.active_pod_id)
                        stage, clean_lines = self.condense_runpod_logs(raw_logs)
                        detailed_stage = f"{stage} ({elapsed_sec}s verlopen)" if "verlopen" not in stage else stage
                        await self.broadcast_boot_telemetry(self.active_pod_id, detailed_stage, last_status or desired_status, clean_lines)
                    except Exception as te:
                        print(f"[RunPod Boot Telemetry] Error updating logs during retry: {te}")

                await asyncio.sleep(4)
