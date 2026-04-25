import asyncio
import json
import numpy as np
import string
import os
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn
from faster_whisper import WhisperModel
import ollama

app = FastAPI()

# Configurable settings via env
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "small") # small, medium, large-v3
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "aya-expanse:8b")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Load model globally on server boot (uses CUDA if available)
print(f"Loading faster-whisper model: {WHISPER_MODEL}")
model = WhisperModel(WHISPER_MODEL, device="cuda", compute_type="float16")
print("Whisper model loaded.")

import logging
logging.basicConfig(level=logging.INFO)

# Global Client for Ollama
client = ollama.AsyncClient(host=OLLAMA_URL)

# --- Automatic Shutdown Timer ---
# Default to 90 minutes (5400 seconds) to prevent runaway costs
MAX_RUN_TIME = int(os.getenv("MAX_RUN_TIME_SEC", "5400"))

def auto_shutdown():
    import time
    import os
    import json
    import urllib.request
    
    print(f"SHUTDOWN TIMER: Server will automatically shut down in {MAX_RUN_TIME/60:.1f} minutes.")
    time.sleep(MAX_RUN_TIME)
    print(f"SHUTDOWN TIMER: Time limit reached ({MAX_RUN_TIME}s). Attempting to stop pod...")
    
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

import threading
threading.Thread(target=auto_shutdown, daemon=True).start()

async def translate_text(text: str, target_lang: str, source_lang: str):
    if not text or not target_lang:
        return ""
        
    source_code = source_lang.split("-")[0] if source_lang else "nl"
    if target_lang == source_code:
        return text

    prompt = (
        f"Translate the following text from {source_code} to {target_lang}. "
        "Do not explain, do not add quotes, just return the translated text.\n\n"
        f"Text to translate:\n{text}"
    )
    
    try:
        response = await client.generate(
            model=OLLAMA_MODEL,
            prompt=prompt,
            options={'temperature': 0}
        )
        t = response['response'].strip()
        t = t.strip('"').strip("'").strip()
        return t
    except Exception as e:
        logging.error(f"Translation Error: {e}")
        return f"[Translation Error: {e}]"


@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    logging.info("Desktop Client connected to RunPod Server")
    
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
                except Exception as e:
                    logging.error(f"JSON parse error: {e}")
                    
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                if len(audio_bytes) > 0:
                    audio_np = np.frombuffer(audio_bytes, dtype=np.float32)
                    
                    # 1. Transcribe with context
                    source_code = config["source_lang"].split("-")[0]
                    transcribe_args = {
                        "language": source_code,
                        "task": "transcribe"
                    }
                    
                    if last_transcript:
                        transcribe_args["initial_prompt"] = last_transcript
                        
                    segments, info = model.transcribe(audio_np, **transcribe_args)
                    text = "".join(segment.text for segment in segments).strip()
                    
                    # Anti-Hallucination
                    if text:
                        clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower()
                        words = clean_text.split()
                        
                        if last_transcript:
                            clean_last = last_transcript.translate(str.maketrans('', '', string.punctuation)).lower()
                            last_words = clean_last.split()
                            
                            if clean_text == clean_last:
                                text = ""
                            elif len(words) >= 3 and clean_text in clean_last:
                                text = ""
                            else:
                                max_overlap = min(len(last_words), len(words))
                                best_overlap = 0
                                for i in range(max_overlap, 0, -1):
                                    if last_words[-i:] == words[:i]:
                                        best_overlap = i
                                        break
                                if best_overlap > 0:
                                    text = " ".join(text.split()[best_overlap:])
                                    
                    if not text:
                        continue
                        
                    # Save context 
                    last_transcript = text[-200:]
                    
                    import time
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
                        
                    await asyncio.gather(*tasks)

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Unexpected WS Error: {e}")

if __name__ == "__main__":
    uvicorn.run("runpod_server:app", host="0.0.0.0", port=8000, reload=False)
