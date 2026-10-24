"""Live Subtitles & Translation ('Ondertitels') - Orchestration Entry Point.

Coordinates FastAPI web server, WebSocket broadcasting, Zeroconf mDNS,
and the modular core transcription and translation engines.
"""

import asyncio
import json
import multiprocessing
import os
import signal
import socket
import threading
import time
import urllib.request
import webbrowser
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from core import (
    AudioStream,
    TranscriptionEngine,
    load_environment,
    resource_path,
)

# Initialize environment configuration
load_environment()

app = FastAPI(title="Ondertitels Live Subtitles")


class ConnectionManager:
    """Manages active WebSocket client connections and broadcasting."""

    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict) -> None:
        dead_connections = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            try:
                self.active_connections.remove(dead)
            except ValueError:
                pass


manager = ConnectionManager()
engine = TranscriptionEngine(manager.broadcast)


# ============================================================================
# FastAPI Lifespan Events
# ============================================================================
@app.on_event("startup")
async def startup_event():
    print("--- NEW APP VERSION STARTED (Async Arch) ---")
    asyncio.create_task(engine.run())

    active_port = int(os.getenv("ACTIVE_PORT", "8000"))

    # Zeroconf (mDNS) Broadcast
    def start_zeroconf():
        try:
            from zeroconf import ServiceInfo, Zeroconf

            time.sleep(1)

            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(('10.255.255.255', 1))
                local_ip = s.getsockname()[0]
            except Exception:
                local_ip = '127.0.0.1'
            finally:
                s.close()

            info = ServiceInfo(
                "_http._tcp.local.",
                "Ondertitels._http._tcp.local.",
                addresses=[socket.inet_aton(local_ip)],
                port=active_port,
                properties={'path': '/live'},
                server="ondertitels.local.",
            )

            app.state.zeroconf = Zeroconf()
            try:
                app.state.zeroconf.register_service(info, allow_name_change=True)
            except Exception as z_err:
                print(f"Notice: Zeroconf registration skipped or name collision: {z_err}")
            app.state.zeroconf_info = info

            mac_name = socket.gethostname()
            if not mac_name.endswith('.local'):
                mac_name += '.local'

            print("\n" + "=" * 60)
            print("🌍 LIVE VIEWER IS READY! Verbinden vanaf telefoon/tablet:")
            print("=" * 60)
            if active_port == 80:
                print("1. http://ondertitels.local/live   (Directe naam via Zeroconf)")
                print(f"2. http://{mac_name}/live       (Apple Bonjour / Native)")
                print(f"3. http://{local_ip}/live            (Direct IP - Als namen niet werken)")
            else:
                print(f"1. http://ondertitels.local:{active_port}/live   (Directe naam via Zeroconf)")
                print(f"2. http://{mac_name}:{active_port}/live       (Apple Bonjour / Native)")
                print(f"3. http://{local_ip}:{active_port}/live            (Direct IP - Als namen niet werken)")
            print("=" * 60 + "\n")

        except Exception as e:
            import traceback
            err_msg = traceback.format_exc()
            print(f"Zeroconf setup failed:\n{err_msg}")

    threading.Thread(target=start_zeroconf, daemon=True).start()

    def open_browser():
        target_url = f"http://127.0.0.1:{active_port}"
        for _ in range(120):
            try:
                with urllib.request.urlopen(target_url, timeout=1.0) as resp:
                    if resp.status == 200:
                        break
            except Exception:
                pass
            time.sleep(0.5)
        webbrowser.open(target_url)

    threading.Thread(target=open_browser, daemon=True).start()


@app.on_event("shutdown")
def shutdown_event():
    try:
        if hasattr(app.state, 'zeroconf'):
            app.state.zeroconf.unregister_service(app.state.zeroconf_info)
            app.state.zeroconf.close()
            print("Zeroconf service unregistered.")
    except Exception:
        pass

    if hasattr(engine.stt_engine, "stop_runpod_pod") and getattr(engine.stt_engine, "active_pod_id", None):
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(engine.stt_engine.stop_runpod_pod())
            else:
                loop.run_until_complete(engine.stt_engine.stop_runpod_pod())
        except Exception as e:
            print(f"[Shutdown] Notice: could not stop pod synchronously: {e}")

    engine.audio_stream.stop(terminate_pyaudio=True)


# ============================================================================
# HTTP & WebSocket Endpoints
# ============================================================================
@app.get("/", response_class=HTMLResponse)
async def get_index():
    with open(resource_path("templates/index.html"), "r", encoding="utf-8") as f:
        return f.read()


@app.get("/live", response_class=HTMLResponse)
@app.get("/live/", response_class=HTMLResponse)
@app.get("/viewer", response_class=HTMLResponse)
async def get_viewer():
    try:
        path = resource_path("templates/viewer.html")
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return HTMLResponse(content=f"Error loading viewer.html: {e}", status_code=500)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        await websocket.send_json({
            "action": "init_config",
            "enable_trans_3": getattr(engine, 'enable_trans_3', False),
            "enable_trans_4": getattr(engine, 'enable_trans_4', False),
            "source_lang": engine.source_lang,
            "target_lang_1": engine.target_lang_1,
            "target_lang_2": engine.target_lang_2,
            "target_lang_3": engine.target_lang_3,
            "target_lang_4": engine.target_lang_4,
            "device_index": engine.audio_stream.current_device_index
        })
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
                if msg.get("action") == "update_config":
                    engine.update_config(msg)
                    await manager.broadcast({
                        "action": "init_config",
                        "enable_trans_3": getattr(engine, 'enable_trans_3', True),
                        "enable_trans_4": getattr(engine, 'enable_trans_4', True)
                    })
                elif msg.get("action") == "get_devices":
                    devices = engine.get_input_devices()
                    await websocket.send_json({"action": "device_list", "devices": devices})
                elif msg.get("action") == "shutdown":
                    print("Shutdown requested from UI")
                    engine.is_paused = True
                    engine.restart_required = True
                    engine.audio_stream.stop(terminate_pyaudio=True)

                    def force_exit():
                        time.sleep(1)
                        print("Forcing exit...")
                        os._exit(0)

                    threading.Thread(target=force_exit, daemon=True).start()
                    os.kill(os.getpid(), signal.SIGINT)
            except Exception as e:
                print(f"WS Handling Error: {e}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Application Entry Point
# ============================================================================
if __name__ == "__main__":
    import uvicorn

    multiprocessing.freeze_support()

    ACTIVE_PORT = 8000
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('0.0.0.0', 80))
        s.close()
        ACTIVE_PORT = 80
    except OSError:
        print("Poort 80 vereist beheerdersrechten (sudo). Terugvallen op poort 8000.")

    os.environ["ACTIVE_PORT"] = str(ACTIVE_PORT)
    print(f"Starting server on http://0.0.0.0:{ACTIVE_PORT}")
    uvicorn.run(app, host="0.0.0.0", port=ACTIVE_PORT)
