import asyncio
import json
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

app = FastAPI()

@app.websocket("/stream")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Local client connected to MOCK RunPod Server")
    
    config = {
        "source_lang": "nl-NL",
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
                        print(f"Server config updated: {config}")
                except Exception as e:
                    print("JSON parse error:", e)
                    
            elif "bytes" in message:
                audio_bytes = message["bytes"]
                # Simulated float32 numpy array
                if len(audio_bytes) > 0:
                    arr = np.frombuffer(audio_bytes, dtype=np.float32)
                    duration = len(arr) / 16000.0
                    print(f"Received {len(audio_bytes)} bytes ({duration:.2f} seconds of audio)")
                    
                    # Simulate processing delay
                    await asyncio.sleep(min(1.0, duration * 0.3))
                    
                    # 1. Send Source return
                    source_stt = f"[MOCK NVIDIA STT] Heard {duration:.1f}s of audio"
                    await websocket.send_json({
                        "action": "update_source",
                        "text": source_stt,
                        "is_final": True
                    })
                    
                    # 2. Simulate Ollama Translation Time
                    await asyncio.sleep(0.5)
                    await websocket.send_json({
                        "action": "update_translation",
                        "col_key": "trans1",
                        "text": f"[MOCK {config.get('target_lang_1')}] Heard {duration:.1f}s"
                    })
                    
                    await websocket.send_json({
                        "action": "update_translation",
                        "col_key": "trans2",
                        "text": f"[MOCK {config.get('target_lang_2')}] Heard {duration:.1f}s"
                    })
                    
                    if config.get("enable_trans_3"):
                        await websocket.send_json({
                            "action": "update_translation",
                            "col_key": "trans3",
                            "text": f"[MOCK {config.get('target_lang_3')}] Heard {duration:.1f}s"
                        })
                        
                    if config.get("enable_trans_4"):
                        await websocket.send_json({
                            "action": "update_translation",
                            "col_key": "trans4",
                            "text": f"[MOCK {config.get('target_lang_4')}] Heard {duration:.1f}s"
                        })
    except WebSocketDisconnect:
        print("Client disconnected.")

if __name__ == "__main__":
    uvicorn.run("mock_runpod:app", host="0.0.0.0", port=8888, reload=True)
