import asyncio
import difflib
import glob
import json
import logging
import os
import shutil
import re
import string
import threading
import time
from typing import Dict, Optional, Set
import urllib.error
import urllib.request

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from faster_whisper import WhisperModel
import ollama

try:
    from core.stt.local_engine import (
        TAIL_HALLUCINATION_PATTERN,
        NOISE_HALLUCINATION_PHRASES,
        BROADCAST_CREDIT_PATTERN,
        filter_anti_hallucination,
    )
except ImportError:
    # Fallback to local definitions if core is unavailable
    NOISE_HALLUCINATION_PHRASES: Set[str] = {
        # Dutch polite closings / noise hallucinations
        "dank u wel",
        "dank u",
        "dank je wel",
        "dank je",
        "hartelijk dank",
        "bedankt",
        "bedankt voor het kijken",
        "bedankt voor het luisteren",
        "tot ziens",
        "fijne dag",
        "fijne avond",
        "dag",
        # English polite closings / noise hallucinations
        "thank you",
        "thank you very much",
        "thanks",
        "thanks for watching",
        "thanks for listening",
        "bye",
        "goodbye",
        # TV / Subtitle credits
        "subtitles by",
        "captioned by",
        "amaraorg",
        "opensubtitles",
        "tv gelderland",
        "omroep gelderland",
        "rtv oost",
        "rtv noord",
        "omroep brabant",
    }

    BROADCAST_CREDIT_PATTERN = re.compile(
        r"^(?:tv|omroep|rtv)\s+[a-z]+(?:\s+\d{4})?$|"
        r"^(?:npo|rtl|nos)(?:\s+\d+)?(?:\s+\d{4})?$|"
        r"^(?:ondertiteling|ondertitels|ondertiteld(?:\s+door)?|subtitles(?:\s+by)?|captioned(?:\s+by)?).*$",
        re.IGNORECASE
    )

    TAIL_HALLUCINATION_PATTERN = re.compile(
        r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)"
        r"(?:"
        r"(?:tv|omroep|rtv)\s+[a-z]+(?:\s+\d{4})?|"
        r"(?:npo|rtl|nos)(?:\s+\d+)?(?:\s+\d{4})?|"
        r"ondertiteling(?:\s+door\s+[a-z]+)?|"
        r"ondertitels|"
        r"bedankt voor het kijken|"
        r"bedankt voor het luisteren|"
        r"thanks for watching|"
        r"thanks for listening|"
        r"subtitles by\s+[a-z0-9\.]+|"
        r"captioned by\s+[a-z0-9\.]+"
        r")[\.\,\!\?\s]*$",
        re.IGNORECASE
    )

    def filter_anti_hallucination(
        text: Optional[str],
        last_transcript: str = "",
        glossary_phrases: Optional[list] = None,
        is_same_segment: bool = False,
    ) -> Optional[str]:
        if not text:
            return None
        cleaned_tail = TAIL_HALLUCINATION_PATTERN.sub("", text).strip()
        if not cleaned_tail:
            return None
        if cleaned_tail != text.strip():
            text = cleaned_tail

        clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
        clean_text = " ".join(clean_text.split())
        words = clean_text.split()
        if not words:
            return None

        clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
        if (
            clean_text in NOISE_HALLUCINATION_PHRASES
            or clean_no_digits in NOISE_HALLUCINATION_PHRASES
            or BROADCAST_CREDIT_PATTERN.match(clean_text)
            or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
        ):
            return None
        elif last_transcript and not is_same_segment:
            clean_last = last_transcript.translate(str.maketrans('', '', string.punctuation)).lower()
            last_words = clean_last.split()
            if clean_text == clean_last:
                return None
            if len(words) >= 3 and clean_text in clean_last:
                return None

            matcher = difflib.SequenceMatcher(None, last_words, words)
            match = matcher.find_longest_match(0, len(last_words), 0, len(words))

            is_large_leak = (match.size >= 4 and match.b <= 5) or (match.size >= 6 and match.b <= 15)
            is_tail_overlap = (match.size >= 1 and match.b == 0 and match.a + match.size == len(last_words))
            is_tail_sub = (match.size >= 2 and match.b <= 5 and match.a + match.size == len(last_words))

            if is_large_leak or is_tail_overlap or is_tail_sub:
                orig_words = text.split()
                cut_idx = match.b + match.size
                trimmed_words = orig_words[cut_idx:]
                text = " ".join(trimmed_words).strip()
                if not text:
                    return None

                clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                clean_text = " ".join(clean_text.split())
                words = clean_text.split()
                if not words:
                    return None

                clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
                if (
                    clean_text in NOISE_HALLUCINATION_PHRASES
                    or clean_no_digits in NOISE_HALLUCINATION_PHRASES
                    or BROADCAST_CREDIT_PATTERN.match(clean_text)
                    or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
                ):
                    return None

        return text

app = FastAPI()

# Temporary directory and cache isolation (prevent writing small files/cache to FUSE /workspace)
os.environ.setdefault("TMPDIR", "/tmp")
os.environ.setdefault("HF_HOME", "/tmp/hf_cache")
os.environ.setdefault("TORCH_HOME", "/tmp/torch_cache")
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("OLLAMA_MODELS", "/root/.ollama/models" if os.path.isdir("/root/.ollama/models") else "/workspace/models/ollama")

# Ensure volatile local SSD cache directories exist
os.makedirs("/tmp/hf_cache", exist_ok=True)
os.makedirs("/tmp/torch_cache", exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Configurable settings via env
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small") # small, medium, large-v3, or path
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya-expanse:8b")
OLLAMA_MODELS = os.getenv("OLLAMA_MODELS", "/root/.ollama/models" if os.path.isdir("/root/.ollama/models") else "/workspace/models/ollama")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

def resolve_whisper_model_path(model_name_or_path: str) -> Optional[str]:
    """Resolve local flat model directory in /root/models, /workspace/models or filesystem if present."""
    base_name = os.path.basename(model_name_or_path.rstrip("/"))
    candidates = []

    # 1. Local high-speed NVMe SSD paths (/root/models)
    candidates.append(os.path.join("/root/models", base_name))
    candidates.append(os.path.join("/root/models", f"faster-whisper-{base_name}"))
    if not os.path.isabs(model_name_or_path):
        candidates.append(os.path.join("/root/models", model_name_or_path))
        candidates.append(os.path.join("/root/models", f"faster-whisper-{model_name_or_path}"))

    # 2. Direct path if valid directory
    if os.path.isdir(model_name_or_path):
        candidates.append(model_name_or_path)

    # 3. Network Global Volume paths (/workspace/models)
    candidates.append(os.path.join("/workspace/models", base_name))
    candidates.append(os.path.join("/workspace/models", f"faster-whisper-{base_name}"))
    if not os.path.isabs(model_name_or_path):
        candidates.append(os.path.join("/workspace/models", model_name_or_path))
        candidates.append(os.path.join("/workspace/models", f"faster-whisper-{model_name_or_path}"))

    # Prefer directories with model.bin present
    for c in candidates:
        if os.path.isdir(c) and os.path.isfile(os.path.join(c, "model.bin")):
            return c

    # Fallback to any directory candidate that exists
    for c in candidates:
        if os.path.isdir(c):
            return c

    return None

def load_whisper_model(model_name: str, device: str = "cuda", compute_type: str = "float16"):
    """Load Faster-Whisper model from flat directory if present, else fallback with self-healing."""
    t0 = time.time()
    resolved_dir = resolve_whisper_model_path(model_name)
    if resolved_dir:
        logging.info(f"[Whisper Init] Loading faster-whisper from local directory: {resolved_dir} (local_files_only=True)...")
        m = WhisperModel(resolved_dir, device=device, compute_type=compute_type, local_files_only=True)
        logging.info(f"[Whisper Init] Whisper model loaded into GPU in {time.time() - t0:.2f}s from local disk!")
        return m

    try:
        logging.warning(f"[Whisper Init] Local model NOT found! Downloading '{model_name}' from web (HuggingFace)...")
        m = WhisperModel(model_name, device=device, compute_type=compute_type)
        logging.info(f"[Whisper Init] Whisper model downloaded and loaded in {time.time() - t0:.2f}s.")
        return m
    except Exception as e:
        logging.warning(f"[Whisper Init] Failed to load faster-whisper model ({e}). Purging corrupted model cache...")
        cache_patterns = [
            "/tmp/hf_cache/hub/models--Systran--faster-whisper*",
            os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper*"),
            "/workspace/huggingface/hub/models--Systran--faster-whisper*",
        ]
        purged = False
        for pattern in cache_patterns:
            for p in glob.glob(pattern):
                logging.info(f"[Whisper Init] Purging corrupted cache: {p}")
                shutil.rmtree(p, ignore_errors=True)
                purged = True
        if purged:
            logging.info("[Whisper Init] Corrupted cache purged successfully. Retrying fresh model download...")
            return WhisperModel(model_name, device=device, compute_type=compute_type)
        raise

model = load_whisper_model(WHISPER_MODEL, device="cuda", compute_type="float16")
logging.info("Whisper model initialized and ready.")

def warmup_ollama(model_name: str, host: str = "http://127.0.0.1:11434", max_retries: int = 2):
    """Pre-warm Ollama LLM into GPU VRAM so subsequent translation requests are instantaneous."""
    for attempt in range(max_retries):
        logging.info(f"[Ollama Init] Pre-warming Ollama {model_name} in GPU VRAM (attempt {attempt + 1}/{max_retries})...")
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{host}/api/chat",
                data=json.dumps({
                    "model": model_name,
                    "messages": [{"role": "user", "content": "hi"}],
                    "stream": False,
                    "options": {"temperature": 0}
                }).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=300) as resp:
                if resp.status == 200:
                    logging.info(f"[Ollama Init] Ollama {model_name} pre-warmed in GPU VRAM in {time.time() - t0:.2f}s!")
                    return True
        except Exception as e:
            logging.warning(f"[Ollama Init] Pre-warming attempt {attempt + 1} notice: {e}")
            if attempt < max_retries - 1:
                time.sleep(3.0)
    logging.error(f"[Ollama Init] Warning: Ollama {model_name} could not be pre-warmed within timeout.")
    return False

warmup_ollama(OLLAMA_MODEL, OLLAMA_URL)
logging.info("Ollama model initialized and ready.")

# Global Client for Ollama
client = ollama.AsyncClient(host=OLLAMA_URL, timeout=120.0)

# --- Automatic Shutdown Timer (Activity-Based) ---
# Default to 90 minutes (5400 seconds) to prevent runaway costs
MAX_RUN_TIME = int(os.getenv("MAX_RUN_TIME_SEC", "5400"))

last_activity_time = time.time()
last_logged_activity = 0.0

def reset_session_timer(reason: str = "activity"):
    global last_activity_time, last_logged_activity
    last_activity_time = time.time()
    last_logged_activity = last_activity_time
    logging.info(f"SHUTDOWN TIMER: Reset triggered ({reason}). Auto-shutdown postponed by {MAX_RUN_TIME/60:.1f} minutes.")

@app.get("/health")
def health_check():
    return {"status": "ok", "whisper_ready": True}

@app.get("/reset_timer")
@app.post("/reset_timer")
async def api_reset_timer():
    reset_session_timer("HTTP API call")
    return {"status": "ok", "message": f"Timer reset. Next shutdown in {MAX_RUN_TIME/60:.1f} minutes."}

def auto_shutdown():
    print(f"SHUTDOWN TIMER: Active. Server will shut down after {MAX_RUN_TIME/60:.1f} minutes of inactivity.")
    
    while True:
        time.sleep(15)
        elapsed = time.time() - last_activity_time
        if elapsed >= MAX_RUN_TIME:
            print(f"SHUTDOWN TIMER: Inactivity limit reached ({int(elapsed)}s). Attempting to stop pod...")
            break
            
    # RunPod injects RUNPOD_POD_ID automatically. 
    # The user must provide RUNPOD_API_KEY in the environment variables.
    api_key = os.getenv("RUNPOD_API_KEY", "").strip()
    pod_id = os.getenv("RUNPOD_POD_ID", "").strip()
    
    if api_key and pod_id:
        terminated = False
        # 1. Primary: REST API v2 DELETE https://api.runpod.io/v2/pods/{pod_id}
        rest_url = f"https://api.runpod.io/v2/pods/{pod_id}"
        rest_headers = {
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        }
        for attempt in range(2):
            try:
                req = urllib.request.Request(rest_url, headers=rest_headers, method="DELETE")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    status = getattr(resp, "status", getattr(resp, "code", 200))
                    if status in (200, 202, 204) or (200 <= status < 300):
                        print(f"SHUTDOWN TIMER: Successfully stopped pod {pod_id} via REST v2 (status {status}).")
                        terminated = True
                        break
            except urllib.error.HTTPError as he:
                if he.code == 429 and attempt == 0:
                    retry_after = he.headers.get("Retry-After") if hasattr(he, "headers") and he.headers else None
                    wait_s = 2.0
                    if retry_after:
                        try:
                            wait_s = float(retry_after)
                        except (ValueError, TypeError):
                            wait_s = 2.0
                    print(f"SHUTDOWN TIMER: Rate limited (429) on REST v2 DELETE, retrying in {wait_s:.1f}s...")
                    time.sleep(wait_s)
                    continue
                print(f"SHUTDOWN TIMER: REST v2 DELETE failed (HTTP {he.code}). Trying GraphQL fallback...")
                break
            except Exception as e:
                print(f"SHUTDOWN TIMER: REST v2 DELETE error: {e}. Trying GraphQL fallback...")
                break

        # 2. Fallback: GraphQL podTerminate
        if not terminated:
            try:
                url = "https://api.runpod.io/graphql"
                mutation = f"""
                mutation {{
                    podTerminate(input: {{podId: "{pod_id}"}})
                }}
                """
                data = json.dumps({"query": mutation}).encode("utf-8")
                headers = {
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {api_key}",
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
                }
                req = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(req, timeout=10) as response:
                    print("RunPod Terminate API Response (GraphQL fallback):", response.read().decode())
            except Exception as e:
                print(f"Failed to call RunPod API: {e}")
    else:
        print("WARNING: RUNPOD_API_KEY or RUNPOD_POD_ID not found in environment.")
        print("The server will exit, but RunPod might restart it automatically. Please add RUNPOD_API_KEY to your template.")
        
    os._exit(0) # Forcefully exit the process as a fallback

threading.Thread(target=auto_shutdown, daemon=True).start()

try:
    from core.translation import clean_translation_meta, LANG_NAMES
except ImportError:
    # Fallback to local definitions if core is unavailable
    LANG_NAMES = {
        "nl": "Dutch", "en": "English", "de": "German", "ru": "Russian",
        "fr": "French", "es": "Spanish", "it": "Italian", "uk": "Ukrainian"
    }

    META_PATTERNS = [
        # English
        r'(?i)\bthe word\s+["\'«„\w]+.*?\btranslates\b.*',
        r'(?i)\bhowever,?\s+without\s+context\b.*',
        r'(?i)\bplease\s+provide\s+(?:the\s+)?(?:full|more|additional)\b.*',
        r'(?i)\bwithout\s+(?:more\s+)?context\b.*',
        r'(?i)\bas\s+per\s+your\s+strict\s+requirements\b.*',
        r'(?i)\bfor\s+example,?\s+if\s+the\s+full\s+sentence\b.*',
        # French
        r'(?i)\ble mot\s+[«"\'\w]+.*?\bse traduit\b.*',
        r'(?i)\bcependant,?\s+sans\s+contexte\b.*',
        r'(?i)\bsans\s+(?:plus\s+de\s+)?contexte\b.*',
        r'(?i)\bveuillez\s+fournir\b.*',
        r'(?i)\bselon\s+vos\s+exigences\s+strictes\b.*',
        r'(?i)\bpar\s+exemple,?\s+si\s+la\s+phrase\b.*',
        # Dutch
        r'(?i)\bhet woord\s+["\'\w]+.*?\bvertaalt\b.*',
        r'(?i)\bechter,?\s+zonder\s+context\b.*',
        r'(?i)\bzonder\s+(?:nadere\s+)?context\b.*',
        r'(?i)\bgeef\s+alstublieft\b.*',
        r'(?i)\bvolgens\s+uw\s+strikte\s+eisen\b.*',
        r'(?i)\bbijvoorbeeld,?\s+als\s+de\b.*',
        # German
        r'(?i)\bdas wort\s+["\'„\w]+.*?\bübersetzt\b.*',
        r'(?i)\bjedoch,?\s+ohne\s+kontext\b.*',
        r'(?i)\bohne\s+(?:weiteren\s+)?kontext\b.*',
        # Russian
        r'(?i)\bслово\s+[«"\'\w]+.*?\bпереводится\b.*',
        r'(?i)\bоднако,?\s+без\s+контекста\b.*',
        r'(?i)\bбез\s+(?:дополнительного\s+)?контекста\b.*',
        # Generic note markers across languages
        r'(?i)\b(?:note|remarque|hinweis|opmerking|примечание|explanation|explication|erklärung|uitleg)\s*:.*',
        r'(?i)\b(?:here\s+is\s+the\s+translation|voici\s+la\s+traduction|hier\s+is\s+de\s+vertaling|hier\s+ist\s+die\s+übersetzung|вот\s+перевод)\s*:?\s*',
    ]

    def clean_translation_meta(text: str) -> str:
        """Strip conversational disclaimers, lecturing, and metadata from LLM translations."""
        if not text:
            return ""
        cleaned = text.strip()

        # If the LLM output a multi-line response with blank line, translation is paragraph 1
        if "\n\n" in cleaned:
            cleaned = cleaned.split("\n\n")[0].strip()

        # Strip introductory markers
        cleaned = re.sub(
            r'(?i)^(?:here\s+is\s+the\s+translation:?\s*|voici\s+la\s+traduction:?\s*|translation:?\s*|traduction:?\s*|vertaling:?\s*|translated\s+text:?\s*)',
            '',
            cleaned
        ).strip()

        # Find earliest trailing meta-commentary pattern
        earliest_idx = len(cleaned)
        for pat in META_PATTERNS:
            if "here" in pat or "voici" in pat or "hier" in pat:
                continue
            m = re.search(pat, cleaned)
            if m and m.start() < earliest_idx:
                earliest_idx = m.start()

        if earliest_idx < len(cleaned):
            cleaned = cleaned[:earliest_idx].strip()

        # Strip surrounding quotes if wrapped (including French « » and German „ “)
        if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
            cleaned = cleaned[1:-1].strip()
        elif cleaned.startswith('«') and cleaned.endswith('»'):
            cleaned = cleaned[1:-1].strip()
        elif cleaned.startswith('„') and cleaned.endswith('“'):
            cleaned = cleaned[1:-1].strip()

        return cleaned


async def translate_text(text: str, target_lang: str, source_lang: str):
    if not text or "***" in text:
        return text
    if not target_lang:
        return ""
        
    source_code = source_lang.split("-")[0].lower() if source_lang else "nl"
    target_code = target_lang.split("-")[0].lower() if target_lang else "en"
    if target_code == source_code:
        return text

    source_lang_name = LANG_NAMES.get(source_code, source_code)
    target_lang_name = LANG_NAMES.get(target_code, target_lang)

    system_prompt = (
        f"You are an automated real-time speech subtitle translation engine for church services.\n"
        f"Translate from {source_lang_name} to {target_lang_name}.\n\n"
        "CORE DIRECTIVES:\n"
        "1. You are a silent translation machine, not an assistant. You have no personality and never converse.\n"
        f"2. Output ONLY the raw translated text in {target_lang_name}. Absolutely zero explanations, notes, intros, or conversation.\n"
        "3. The input is live spoken audio transcribed in real-time. It frequently contains sentence fragments, pauses, and dangling words (e.g. 'over.', 'en.', 'maar.').\n"
        f"4. When input ends on an incomplete thought or trailing word, translate those exact words directly into {target_lang_name} and stop immediately.\n"
        "5. Never explain words, never comment on missing context or grammar, and never ask for more context in any language.\n"
        "6. Translate accurately and without omissions. Never omit proper names, biblical figures (e.g. David, Jesus, Moses, Paul), locations, numbers, or clauses.\n"
        "7. Do not summarize, condense, or simplify the text.\n"
    )
    
    t_trans_start = time.time()
    try:
        response = await client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            options={'temperature': 0}
        )
        t = response['message']['content'].strip()
        trans_elapsed = time.time() - t_trans_start
        cleaned = clean_translation_meta(t)
        logging.info(f"[Ollama] Translated ({source_code}->{target_code}) in {trans_elapsed:.3f}s: '{cleaned}'")
        return cleaned
    except Exception as e:
        logging.error(f"[Ollama] Translation Error ({source_code}->{target_code}): {e}")
        return text


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Desktop Client connected to RunPod Server")
    reset_session_timer("Desktop Client connected")
    
    # Track prompt context for sliding window
    last_transcript = ""
    current_seg_id = None
    current_status = "final"
    current_sample_width = 4
    last_processed_seg_id = None
    active_chain_tasks: Dict[str, asyncio.Task] = {}
    chain_busy: Dict[str, bool] = {"trans1": False, "trans2": False}
    
    # Default configs
    config = {
        "source_lang": "nl",
        "target_lang_1": "en",
        "target_lang_2": "ru",
        "target_lang_3": "nl",
        "target_lang_4": "nl",
        "enable_trans_3": True,
        "enable_trans_4": True
    }
    
    try:
        while True:
            message = await websocket.receive()
            
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("action") == "config":
                        config.update(data)
                        logging.info(f"Server config updated: {config}")
                        reset_session_timer("Client config received")
                    elif data.get("action") == "audio_meta":
                        current_seg_id = data.get("seg_id")
                        current_status = data.get("status", "final")
                    elif data.get("action") == "audio":
                        current_seg_id = data.get("id") or data.get("seg_id")
                        current_status = data.get("status", "final")
                        if "sample_width" in data:
                            current_sample_width = int(data["sample_width"])
                    elif data.get("action") == "reset_timer":
                        reset_session_timer("Client heartbeat/timer reset")
                    elif data.get("action") == "probe":
                        probe_id = data.get("id", f"probe_{int(time.time()*1000)}")
                        reset_session_timer("Heartbeat probe")
                        await websocket.send_json({
                            "id": probe_id,
                            "type": "probe_response",
                            "status": "ok"
                        })
                except Exception as e:
                    logging.error(f"JSON parse error: {e}")
                    
            elif "bytes" in message:
                try:
                    audio_bytes = message["bytes"]
                    if len(audio_bytes) > 0:
                        global last_activity_time, last_logged_activity
                        last_activity_time = time.time()
                        if time.time() - last_logged_activity > 60:
                            last_logged_activity = time.time()
                            logging.info(f"SHUTDOWN TIMER: Active audio streaming. Auto-shutdown postponed by {MAX_RUN_TIME/60:.1f} minutes.")
                            
                        if current_sample_width == 2:
                            audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                        else:
                            try:
                                parsed = np.frombuffer(audio_bytes, dtype=np.float32)
                                if np.all(np.isfinite(parsed)) and np.max(np.abs(parsed)) <= 5.0:
                                    audio_np = parsed
                                else:
                                    audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                            except Exception:
                                audio_np = np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0
                    
                    seg_id = current_seg_id or f"seg_{int(time.time()*1000)}"
                    status = current_status or "final"
                    is_same_segment = bool(last_processed_seg_id and last_processed_seg_id == seg_id)

                    # 1. Transcribe with context
                    source_code = config["source_lang"].split("-")[0]
                    transcribe_args = {
                        "language": source_code,
                        "task": "transcribe",
                        "temperature": 0.0,
                        "condition_on_previous_text": False,
                    }
                    
                    if last_transcript:
                        transcribe_args["initial_prompt"] = last_transcript
                        
                    t_stt_start = time.time()
                    loop = asyncio.get_running_loop()
                    def _transcribe():
                        segments, info = model.transcribe(audio_np, **transcribe_args)
                        return "".join(segment.text for segment in segments).strip()

                    text = await loop.run_in_executor(None, _transcribe)
                    stt_elapsed = time.time() - t_stt_start
                    audio_dur = len(audio_np) / 16000.0 if len(audio_np) > 0 else 0.0
                    logging.info(f"[Whisper] Transcribed {audio_dur:.2f}s audio in {stt_elapsed:.3f}s: '{text}'")
                    
                    # Anti-Hallucination
                    if text:
                        filtered = filter_anti_hallucination(
                            text,
                            last_transcript=last_transcript,
                            is_same_segment=is_same_segment,
                        )
                        text = filtered or ""

                    if not text:
                        continue
                        
                    last_processed_seg_id = seg_id
                    if status == "final":
                        last_transcript = text[-1200:]
                        current_seg_id = None
                        current_status = "final"
                    
                    # Dispatch to websocket
                    await websocket.send_json({
                        "id": seg_id,
                        "status": status,
                        "is_delta": False,
                        "col1_text": text
                    })
                    
                    # Dispatch parallel translation chains with is_busy skip logic and guaranteed final completion
                    async def process_translation_chain(col_key_1, lang_1, col_key_2, lang_2, segment_id, segment_status, current_text, prev_task):
                        if prev_task and not prev_task.done():
                            try:
                                await asyncio.wait_for(asyncio.shield(prev_task), timeout=3.0)
                            except Exception:
                                pass
                        chain_busy[col_key_1] = True
                        try:
                            # First translation (e.g. NL -> EN / RU)
                            text_1 = await translate_text(current_text, lang_1, config["source_lang"])
                            col_idx_1 = "col2_text" if col_key_1 == "trans1" else "col3_text"
                            
                            await websocket.send_json({
                                "id": segment_id,
                                "status": segment_status,
                                "is_delta": False,
                                col_idx_1: text_1
                            })
                            
                            # Second translation (Back-translation: e.g. EN / RU -> NL) - ONLY on final chunks
                            if col_key_2 and lang_2 and segment_status == "final":
                                text_2 = await translate_text(text_1, lang_2, lang_1)
                                col_idx_2 = "col4_text" if col_key_2 == "trans3" else "col5_text"
                                await websocket.send_json({
                                    "id": segment_id,
                                    "status": segment_status,
                                    "is_delta": False,
                                    col_idx_2: text_2
                                })
                        except Exception as ex:
                            logging.error(f"Error in translation chain {col_key_1}: {ex}", exc_info=True)
                        finally:
                            chain_busy[col_key_1] = False

                    # Chain 1: trans1 -> trans3 (Back-translation 1)
                    if config.get("target_lang_1"):
                        c3_key = "trans3" if config.get("enable_trans_3") else None
                        c3_lang = config.get("target_lang_3") if config.get("enable_trans_3") else None
                        if chain_busy.get("trans1", False):
                            if status == "final":
                                prev1 = active_chain_tasks.get("trans1")
                                active_chain_tasks["trans1"] = asyncio.create_task(
                                    process_translation_chain("trans1", config.get("target_lang_1"), c3_key, c3_lang, seg_id, status, text, prev1)
                                )
                            # else: skip interim chunk while trans1 is busy to prevent socket aborts
                        else:
                            chain_busy["trans1"] = True
                            prev1 = active_chain_tasks.get("trans1")
                            active_chain_tasks["trans1"] = asyncio.create_task(
                                process_translation_chain("trans1", config.get("target_lang_1"), c3_key, c3_lang, seg_id, status, text, prev1)
                            )

                    # Chain 2: trans2 -> trans4 (Back-translation 2)
                    if config.get("target_lang_2"):
                        c4_key = "trans4" if config.get("enable_trans_4") else None
                        c4_lang = config.get("target_lang_4") if config.get("enable_trans_4") else None
                        if chain_busy.get("trans2", False):
                            if status == "final":
                                prev2 = active_chain_tasks.get("trans2")
                                active_chain_tasks["trans2"] = asyncio.create_task(
                                    process_translation_chain("trans2", config.get("target_lang_2"), c4_key, c4_lang, seg_id, status, text, prev2)
                                )
                            # else: skip interim chunk while trans2 is busy to prevent socket aborts
                        else:
                            chain_busy["trans2"] = True
                            prev2 = active_chain_tasks.get("trans2")
                            active_chain_tasks["trans2"] = asyncio.create_task(
                                process_translation_chain("trans2", config.get("target_lang_2"), c4_key, c4_lang, seg_id, status, text, prev2)
                            )
                except Exception as e:
                    logging.error(f"Error processing audio chunk: {e}", exc_info=True)

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Unexpected WS Error: {e}")
    finally:
        for t in active_chain_tasks.values():
            if not t.done():
                t.cancel()

if __name__ == "__main__":
    uvicorn.run(
        "runpod_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_ping_interval=30.0,
        ws_ping_timeout=120.0,
    )
