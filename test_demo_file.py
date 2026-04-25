import pyaudio
import wave
import time
import glob

files = glob.glob("*.wav")
if not files:
    print("No wav files found")
    exit(1)
    
file_path = files[0]
print(f"Testing {file_path}")

p = pyaudio.PyAudio()

try:
    wf = wave.open(file_path, 'rb')
    print("Wave opened")
    out_stream = p.open(
        format=p.get_format_from_width(wf.getsampwidth()),
        channels=wf.getnchannels(),
        rate=wf.getframerate(),
        output=True
    )
    print("Output stream opened")
    
    data = wf.readframes(480)
    if not data:
        print("No data read")
        
    print(f"Read {len(data)} bytes. Success.")
    
    out_stream.stop_stream()
    out_stream.close()
    
except Exception as e:
    print(f"Error: {e}")
finally:
    p.terminate()

