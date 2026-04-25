import pyaudio
import time
import struct
import math

CHUNK = 480
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000

p = pyaudio.PyAudio()

try:
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("* recording 5 seconds test")
    
    for i in range(0, int(RATE / CHUNK * 5)):
        data = stream.read(CHUNK, exception_on_overflow=False)
        
        count = len(data) // 2
        shorts = struct.unpack(f"<{count}h", data)
        sum_squares = sum(s * s for s in shorts)
        rms = int(math.sqrt(sum_squares / count))
        
        print(f"[{i:02d}] RMS Volume: {rms} {'|' * (rms // 10)}")

    print("* done recording")

    stream.stop_stream()
    stream.close()
    
except Exception as e:
    print(f"Error: {e}")
finally:
    p.terminate()
