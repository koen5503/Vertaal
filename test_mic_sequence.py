import pyaudio
import threading
import time

def calculate_rms(data):
    import struct, math
    count = len(data) // 2
    format = "<%dh" % count
    shorts = struct.unpack(format, data)
    sum_squares = sum(s * s for s in shorts)
    return int(math.sqrt(sum_squares / count))

class TestStream:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            input_device_index=None,
            frames_per_buffer=480,
        )
        self.running = False

    def start(self):
        self.running = True
        if self.stream.is_stopped():
            self.stream.start_stream()
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        print("Read loop started")
        while self.running:
            try:
                data = self.stream.read(480, exception_on_overflow=False)
                print(f"Read chunk... RMS: {calculate_rms(data)}")
                time.sleep(0.01)
            except Exception as e:
                print(f"Error in read: {e}")
                break

if __name__ == "__main__":
    t = TestStream()
    print("Waiting 3 seconds like uvicorn startup...")
    time.sleep(3)
    t.start()
    time.sleep(2)
    t.running = False
