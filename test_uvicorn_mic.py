import pyaudio
import threading
import time
from fastapi import FastAPI
import uvicorn
import math
import struct

app = FastAPI()

def calculate_rms(data):
    count = len(data) // 2
    format = "<%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = sum(s * s for s in shorts)
    return int(math.sqrt(sum_squares / count))

p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=480,
)

def _read_loop():
    print("Read loop started")
    while True:
        try:
            data = stream.read(480, exception_on_overflow=False)
            rms = calculate_rms(data)
            if rms > 0:
                print(f"Read chunk... RMS: {rms}")
            else:
                print(f"Read chunk... SILENCE")
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in read: {e}")
            break

@app.on_event("startup")
def startup():
    print("Starting background thread...")
    threading.Thread(target=_read_loop, daemon=True).start()

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8001)
