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
from typing import Set
import urllib.request

import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from faster_whisper import WhisperModel
import ollama

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

app = FastAPI()

# Configurable settings via env
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small") # small, medium, large-v3
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya-expanse:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

def load_whisper_model(model_name: str, device: str = "cuda", compute_type: str = "float16"):
    """Load Faster-Whisper model with self-healing recovery against partial/corrupt downloads."""
    try:
        print(f"Loading faster-whisper model: {model_name}")
        return WhisperModel(model_name, device=device, compute_type=compute_type)
    except Exception as e:
        print(f"Warning: Failed to load faster-whisper model ({e}). Purging corrupted model cache...")
        cache_patterns = [
            "/workspace/huggingface/hub/models--Systran--faster-whisper*",
            os.path.expanduser("~/.cache/huggingface/hub/models--Systran--faster-whisper*"),
        ]
        purged = False
        for pattern in cache_patterns:
            for p in glob.glob(pattern):
                print(f"Purging corrupted cache: {p}")
                shutil.rmtree(p, ignore_errors=True)
                purged = True
        if purged:
            print("Corrupted cache purged successfully. Retrying fresh model download...")
            return WhisperModel(model_name, device=device, compute_type=compute_type)
        raise

model = load_whisper_model(WHISPER_MODEL, device="cuda", compute_type="float16")
print("Whisper model loaded.")

logging.basicConfig(level=logging.INFO)

# Global Client for Ollama
client = ollama.AsyncClient(host=OLLAMA_URL)

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
        try:
            url = "https://api.runpod.io/graphql"
            mutation = f"""
            mutation {{
                podStop(input: {{podId: "{pod_id}"}}) {{
                    id
                    desiredStatus
                }}
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
                print("RunPod Stop API Response:", response.read().decode())
        except Exception as e:
            print(f"Failed to call RunPod API: {e}")
    else:
        print("WARNING: RUNPOD_API_KEY or RUNPOD_POD_ID not found in environment.")
        print("The server will exit, but RunPod might restart it automatically. Please add RUNPOD_API_KEY to your template.")
        
    os._exit(0) # Forcefully exit the process as a fallback

threading.Thread(target=auto_shutdown, daemon=True).start()

LANG_NAMES = {
    "nl": "Dutch", "en": "English", "de": "German", "ru": "Russian",
    "fr": "French", "es": "Spanish", "it": "Italian", "uk": "Ukrainian"
}

META_PATTERNS = [
    r'(?i)\bthe word\s+["\'\w]+(?:\s+in\s+\w+)?\s+translates\b.*',
    r'(?i)\bhowever,?\s+without\s+context\b.*',
    r'(?i)\bplease\s+provide\s+(?:the\s+)?(?:full|more|additional)\b.*',
    r'(?i)\bwithout\s+(?:more\s+)?context\b.*',
    r'(?i)\bas\s+per\s+your\s+strict\s+requirements\b.*',
    r'(?i)\bfor\s+example,?\s+if\s+the\s+full\s+sentence\b.*',
    r'(?i)\bnote:\s*.*',
    r'(?i)\bexplanation:\s*.*',
    r'(?i)\bcontext:\s*.*',
    r'(?i)\bhere\s+is\s+the\s+translation:?\s*',
]


def clean_translation_meta(text: str) -> str:
    """Strip conversational disclaimers, lecturing, and metadata from LLM translations."""
    if not text:
        return ""
    cleaned = text.strip()

    # Strip introductory markers like "Here is the translation:"
    cleaned = re.sub(
        r'(?i)^(?:here\s+is\s+the\s+translation:?\s*|translation:?\s*|translated\s+text:?\s*)',
        '',
        cleaned
    ).strip()

    # Find earliest trailing meta-commentary pattern
    earliest_idx = len(cleaned)
    for pat in META_PATTERNS:
        if "here" in pat:
            continue
        m = re.search(pat, cleaned)
        if m and m.start() < earliest_idx:
            earliest_idx = m.start()

    if earliest_idx < len(cleaned):
        cleaned = cleaned[:earliest_idx].strip()

    # Strip surrounding quotes if wrapped
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
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
        f"You are a real-time speech subtitle translator. Translate from {source_lang_name} to {target_lang_name}.\n\n"
        "CRITICAL TRANSLATION INSTRUCTIONS:\n"
        "- Output ONLY the direct translation. Absolutely zero notes, explanations, disclaimers, or conversation.\n"
        "- Translate accurately, completely, and faithfully without omissions.\n"
        "- Never omit proper names, biblical figures (e.g. David, Jesus, Moses, Paul), locations, numbers, or clauses.\n"
        "- Do not summarize, condense, or simplify the text.\n"
        "- The input is real-time spoken audio and may contain pauses, sentence fragments, or trailing words (e.g. 'over.', 'en.', 'maar.').\n"
        "- Translate incomplete thoughts and trailing words directly and literally as far as they go (e.g. 'over.' -> 'about.' or 'over.').\n"
        "- NEVER explain words or state that context is missing. Never ask for more context.\n"
        "- NEVER output phrases like 'However, without context...', 'The word... translates to...', or 'Please provide'."
    )
    
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
        return clean_translation_meta(t)
    except Exception as e:
        logging.error(f"Translation Error: {e}")
        return f"[Translation Error: {e}]"


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Desktop Client connected to RunPod Server")
    reset_session_timer("Desktop Client connected")
    
    # Track prompt context for sliding window
    last_transcript = ""
    
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
                            
                        audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # 1. Transcribe with context
                    source_code = config["source_lang"].split("-")[0]
                    transcribe_args = {
                        "language": source_code,
                        "task": "transcribe"
                    }
                    
                    if last_transcript:
                        transcribe_args["initial_prompt"] = last_transcript
                        
                    loop = asyncio.get_running_loop()
                    def _transcribe():
                        segments, info = model.transcribe(audio_np, **transcribe_args)
                        return "".join(segment.text for segment in segments).strip()

                    text = await loop.run_in_executor(None, _transcribe)
                    
                    # Anti-Hallucination
                    if text:
                        cleaned_tail = TAIL_HALLUCINATION_PATTERN.sub("", text).strip()
                        if not cleaned_tail:
                            text = ""
                        elif cleaned_tail != text.strip():
                            text = cleaned_tail

                    if text:
                        clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                        clean_text = " ".join(clean_text.split())
                        words = clean_text.split()

                        clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
                        if (
                            clean_text in NOISE_HALLUCINATION_PHRASES
                            or clean_no_digits in NOISE_HALLUCINATION_PHRASES
                            or BROADCAST_CREDIT_PATTERN.match(clean_text)
                            or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
                        ):
                            text = ""
                        elif last_transcript:
                            clean_last = last_transcript.translate(str.maketrans('', '', string.punctuation)).lower()
                            last_words = clean_last.split()
                            
                            if clean_text == clean_last:
                                text = ""
                            elif len(words) >= 3 and clean_text in clean_last:
                                text = ""
                            else:
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

                                if text:
                                    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                                    clean_text = " ".join(clean_text.split())
                                    clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
                                    if (
                                        clean_text in NOISE_HALLUCINATION_PHRASES
                                        or clean_no_digits in NOISE_HALLUCINATION_PHRASES
                                        or BROADCAST_CREDIT_PATTERN.match(clean_text)
                                        or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
                                    ):
                                        text = ""

                    if not text:
                        continue
                        
                    # Save context 
                    last_transcript = text[-1200:]
                    
                    seg_id = f"seg_{int(time.time()*1000)}"
                    
                    # Dispatch to websocket
                    await websocket.send_json({
                        "id": seg_id,
                        "status": "final",
                        "col1_text": text
                    })
                    
                    # Dispatch parallel translation chains
                    async def process_translation_chain(col_key_1, lang_1, col_key_2, lang_2, segment_id):
                        # First translation (e.g. NL -> EN)
                        text_1 = await translate_text(text, lang_1, config["source_lang"])
                        col_idx_1 = "col2_text" if col_key_1 == "trans1" else "col3_text"
                        
                        await websocket.send_json({
                            "id": segment_id,
                            "status": "final",
                            col_idx_1: text_1
                        })
                        
                        # Second translation (Back-translation: EN -> NL)
                        if col_key_2 and lang_2:
                            text_2 = await translate_text(text_1, lang_2, lang_1)
                            col_idx_2 = "col4_text" if col_key_2 == "trans3" else "col5_text"
                            await websocket.send_json({
                                "id": segment_id,
                                "status": "final",
                                col_idx_2: text_2
                            })

                    tasks = []
                    
                    # Chain 1: trans1 -> trans3 (Back-translation 1)
                    c3_key = "trans3" if config.get("enable_trans_3") else None
                    c3_lang = config.get("target_lang_3") if config.get("enable_trans_3") else None
                    tasks.append(process_translation_chain("trans1", config.get("target_lang_1"), c3_key, c3_lang, seg_id))
                    
                    # Chain 2: trans2 -> trans4 (Back-translation 2)
                    c4_key = "trans4" if config.get("enable_trans_4") else None
                    c4_lang = config.get("target_lang_4") if config.get("enable_trans_4") else None
                    tasks.append(process_translation_chain("trans2", config.get("target_lang_2"), c4_key, c4_lang, seg_id))
                        
                    async def _dispatch_translations(task_list):
                        try:
                            await asyncio.gather(*task_list)
                        except Exception as ex:
                            logging.error(f"Error in translation background task: {ex}", exc_info=True)

                    if tasks:
                        asyncio.create_task(_dispatch_translations(tasks))
                except Exception as e:
                    logging.error(f"Error processing audio chunk: {e}", exc_info=True)

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Unexpected WS Error: {e}")

if __name__ == "__main__":
    uvicorn.run(
        "runpod_server:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        ws_ping_interval=30.0,
        ws_ping_timeout=120.0,
    )
