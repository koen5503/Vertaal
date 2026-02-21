import pyaudio
import struct
import math
import time

def calculate_rms(data):
    count = len(data) // 2
    format = "<%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = sum(s * s for s in shorts)
    return int(math.sqrt(sum_squares / count))

p = pyaudio.PyAudio()

print("--- Available Audio Devices ---")
for i in range(p.get_device_count()):
    dev = p.get_device_info_by_index(i)
    if dev['maxInputChannels'] > 0:
        print(f"Index {i}: {dev['name']} (Channels: {dev['maxInputChannels']})")

default_device = p.get_default_input_device_info()
print(f"\nDefault Device: {default_device['name']} (Index: {default_device['index']})")

print("\n--- Testing Recording from Default Device ---")
try:
    stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=16000,
                    input=True,
                    frames_per_buffer=1024)

    print("Recording... (Press Ctrl+C to stop)")
    while True:
        data = stream.read(1024, exception_on_overflow=False)
        rms = calculate_rms(data)
        bars = "#" * int(rms / 50)
        print(f"RMS: {rms:5d} | {bars}")
        time.sleep(0.05)
except KeyboardInterrupt:
    print("\nStopped.")
except Exception as e:
    print(f"\nError: {e}")
finally:
    if 'stream' in locals():
        stream.stop_stream()
        stream.close()
    p.terminate()
