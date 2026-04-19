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
                    
                    # Dispatch to websocket
                    await websocket.send_json({
                        "action": "update_source",
                        "text": text,
                        "is_final": True
                    })
                    
                    # Dispatch parallel translation tasks
                    async def process_translation(col_key, target_lang):
                        res = await translate_text(text, target_lang, config["source_lang"])
                        await websocket.send_json({
                            "action": "update_translation",
                            "col_key": col_key,
                            "text": res
                        })

                    tasks = []
                    tasks.append(process_translation("trans1", config["target_lang_1"]))
                    tasks.append(process_translation("trans2", config["target_lang_2"]))
                    if config["enable_trans_3"]:
                        tasks.append(process_translation("trans3", config["target_lang_3"]))
                    if config["enable_trans_4"]:
                        tasks.append(process_translation("trans4", config["target_lang_4"]))
                        
                    await asyncio.gather(*tasks)

    except WebSocketDisconnect:
        logging.info("Client disconnected.")
    except Exception as e:
        logging.error(f"Unexpected WS Error: {e}")

if __name__ == "__main__":
    uvicorn.run("runpod_server:app", host="0.0.0.0", port=8000, reload=False)
