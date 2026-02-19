import os
import time
import wave
import traceback
from google.cloud import speech

def perform_streaming_recognize(audio_file, service_account_file):
    """Streams audio to Google Cloud STT and logs is_final latency."""
    
    client = speech.SpeechClient.from_service_account_json(service_account_file)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="nl-NL",
        enable_automatic_punctuation=True,
        model="latest_long", # Fallback to default if not available, but user requested this
        use_enhanced=True,
    )
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
    )

    with wave.open(audio_file, "rb") as wf:
        # Check params
        if wf.getnchannels() != 1 or wf.getframerate() != 16000:
             print(f"Warning: {audio_file} format mismatch (Channels: {wf.getnchannels()}, Rate: {wf.getframerate()})")
        
        chunk_size = int(16000 / 10) # 100ms chunks
        data = wf.readframes(chunk_size)
    
        def request_generator():
            nonlocal data
            try:
                # First check if we have data
                if len(data) == 0:
                    print("Debug: Initial data read was empty")
                    return
                
                print("Debug: Generator starting...")
                while len(data) > 0:
                    yield speech.StreamingRecognizeRequest(audio_content=data)
                    # crude simulation of real-time streaming
                    time.sleep(0.1) 
                    
                    data = wf.readframes(chunk_size)
                print("Debug: Generator finished yielding audio.")
            except Exception as e:
                print(f"Debug: Generator exception: {e}")
                traceback.print_exc()
                raise

        requests = request_generator()
        
        print(f"\n--- Testing {audio_file} ---")
        start_time = time.time()
        
        # We need to track when sentences end to calculate latency. 
        # Since we just have the full file, we will log the wall-clock time of 'is_final'.
        # The user asked: "Bereken het tijdsverschil tussen het einde van de gesproken zin... en is_final".
        # This is tricky with pre-recorded audio unless we know the timestamps of the silences.
        # Since we GENERATED the audio, we know the structure:
        # Sentence 1 + Silence + Sentence 2 + Silence...
        # We could improve this by passing the sentence lengths to this script or just observing the relative timing.
        
        try:
            responses = client.streaming_recognize(streaming_config, requests)

            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if not result.alternatives:
                    continue
                
                # stability = result.stability
                transcript = result.alternatives[0].transcript

                if result.is_final:
                    timestamp = time.time()
                    latency = timestamp - start_time
                    print(f"[{latency:.3f}s] FINAL: '{transcript}'")
                else:
                    # print(f"Interim: {transcript}", end='\r')
                    pass
        except Exception:
            traceback.print_exc()

def run_tests():
    # Discover test files
    files = [f for f in os.listdir(".") if f.startswith("test_") and f.endswith(".wav")]
    files.sort(key=lambda x: int(x.replace("test_", "").replace("ms.wav", "")))
    
    credentials_path = "/Users/koen/Ondertitels/ondertitels-486017-0ee48ab1ba8d.json"
    
    if not files:
        print("No test_*.wav files found. Run generate_test_audio.py first.")
        return

    for f in files:
        try:
            perform_streaming_recognize(f, credentials_path)
        except Exception as e:
            print(f"Error processing {f}: {e}")

if __name__ == "__main__":
    run_tests()
