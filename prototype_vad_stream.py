import collections
import contextlib
import wave
import time
import sys
from types import ModuleType

# Mock pkg_resources to satisfy webrtcvad dependency
if 'pkg_resources' not in sys.modules:
    class MockDistribution:
        version = '2.0.10'
    class MockPkgResources:
        @staticmethod
        def get_distribution(name):
            return MockDistribution()
    sys.modules['pkg_resources'] = MockPkgResources()

import webrtcvad
import traceback
from google.cloud import speech

# VAD Parameters
FRAME_DURATION_MS = 30
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_WIDTH = 2 # 16-bit
CHUNK_SIZE_BYTES = int(SAMPLE_RATE * (FRAME_DURATION_MS / 1000.0) * SAMPLE_WIDTH)

class VadAudioSource:
    """Read a WAV file and yield chunks with VAD detection."""
    def __init__(self, filename, aggressiveness=3):
        self.filename = filename
        self.vad = webrtcvad.Vad(aggressiveness)
        self.wf = wave.open(filename, 'rb')
        
        if self.wf.getframerate() != SAMPLE_RATE or self.wf.getnchannels() != CHANNELS:
            raise ValueError("File must be 16kHz Mono PCM.")
            
        self.silence_frames = 0
        self.speech_frames = 0
        self.current_silence_duration_ms = 0
        self.finished = False

    def read_chunk(self):
        data = self.wf.readframes(int(CHUNK_SIZE_BYTES / SAMPLE_WIDTH))
        if len(data) == 0:
            self.finished = True
            return None
        
        # If last chunk is partial, pad it with silence or just return
        if len(data) < CHUNK_SIZE_BYTES:
            # Pad with zeros to meet 30ms requirement for last frame
            missing = CHUNK_SIZE_BYTES - len(data)
            data += b'\x00' * missing

        # Apply VAD
        is_speech = self.vad.is_speech(data, SAMPLE_RATE)
        
        if is_speech:
            self.silence_frames = 0
            self.speech_frames += 1
            self.current_silence_duration_ms = 0
        else:
            self.silence_frames += 1
            self.current_silence_duration_ms = self.silence_frames * FRAME_DURATION_MS

        return data
    
    def close(self):
        self.wf.close()

class StreamResetHandler:
    """Manages the Google Cloud Speech Stream and handles resets."""
    def __init__(self, audio_source):
        self.audio_source = audio_source
        self.client = speech.SpeechClient.from_service_account_json(
            "/Users/koen/Ondertitels/ondertitels-486017-0ee48ab1ba8d.json"
        )
        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code="nl-NL",
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
        )

    def generator(self, stop_flag):
        """Yields audio chunks to Google API until stop_flag is set."""
        while not self.audio_source.finished and not stop_flag['stop']:
            chunk = self.audio_source.read_chunk()
            if chunk is None:
                break
            
            yield speech.StreamingRecognizeRequest(audio_content=chunk)
            
            # Simulate real-time
            time.sleep(FRAME_DURATION_MS / 1000.0)

    def run_loop(self):
        """Main loop that restarts streams based on VAD/Stability."""
        sentence_count = 0
        
        while not self.audio_source.finished:
            print(f"\n--- Starting Stream {sentence_count + 1} ---")
            stop_flag = {'stop': False}
            
            requests = self.generator(stop_flag)
            
            try:
                responses = self.client.streaming_recognize(
                    config=self.streaming_config, 
                    requests=requests
                )

                for response in responses:
                    if not response.results:
                        continue
                        
                    result = response.results[0]
                    if not result.alternatives:
                        continue
                        
                    transcript = result.alternatives[0].transcript
                    stability = result.stability
                    is_final = result.is_final
                    
                    silence_ms = self.audio_source.current_silence_duration_ms
                    
                    # LOGGING
                    # Start of line refresh
                    sys.stdout.write(f"\rSilence: {silence_ms}ms | Stability: {stability:.2f} | Text: {transcript[:50]}...")
                    sys.stdout.flush()

                    # CUT CONDITION
                    # 1. Silence > 400ms
                    # 2. Stability > 0.8
                    if silence_ms > 400 and stability > 0.8 and not is_final:
                        print(f"\n[FORCING CUT] VAD Silence: {silence_ms}ms | Stability: {stability:.2f}")
                        print(f"FINALIZED (Local): {transcript}")
                        stop_flag['stop'] = True
                        break # Break response loop, generator will see flag and stop
                    
                    # If Google finalizes it naturally (unexpected with this model/latency, but possible)
                    if is_final:
                        print(f"\n[GOOGLE FINAL] {transcript}")
                        stop_flag['stop'] = True
                        break

            except Exception as e:
                print(f"\nStream Error: {e}")
                # If error, break to restart
                traceback.print_exc()
                stop_flag['stop'] = True
            
            sentence_count += 1
            if self.audio_source.finished:
                print("\nAudio finished.")

def run_prototype():
    # Test on one of the files from Phase 2
    filename = "test_400ms.wav" 
    # Or test_1000ms.wav which we know fails on standard model
    
    print("Running Client-Side Segmentation Prototype Check...")
    files = ["test_400ms.wav", "test_1000ms.wav"]
    
    for f in files:
        print(f"\n\n{'='*30}\nTesting File: {f}\n{'='*30}")
        try:
            source = VadAudioSource(f)
            handler = StreamResetHandler(source)
            handler.run_loop()
            source.close()
        except Exception as e:
            print(f"Failed to run on {f}: {e}")

if __name__ == "__main__":
    run_prototype()
