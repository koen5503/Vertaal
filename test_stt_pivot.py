import os
import time
import wave
import traceback
from google.cloud import speech

def run_stt_test(test_name, audio_files, config_overrides):
    """
    Runs a specific STT configuration test on a list of audio files.
    """
    print(f"\n{'='*20} Running {test_name} {'='*20}")
    
    credentials_path = "/Users/koen/Ondertitels/ondertitels-486017-0ee48ab1ba8d.json"
    client = speech.SpeechClient.from_service_account_json(credentials_path)

    base_config = {
        "encoding": speech.RecognitionConfig.AudioEncoding.LINEAR16,
        "sample_rate_hertz": 16000,
        "language_code": "nl-NL",
        "enable_automatic_punctuation": True,
        "model": "default", # Default base model
        "use_enhanced": True,
    }
    
    # Apply overrides
    final_config_dict = base_config.copy()
    final_config_dict.update(config_overrides)
    
    # Extract special tracking flags that aren't part of RecognitionConfig
    single_utterance = final_config_dict.pop("single_utterance", False)
    log_interim_stability = final_config_dict.pop("log_interim_stability", False)

    config = speech.RecognitionConfig(**final_config_dict)
    
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=single_utterance
    )

    for audio_file in audio_files:
        print(f"\n--- Testing {audio_file} ---")
        
        with wave.open(audio_file, "rb") as wf:
            chunk_size = int(16000 / 10) # 100ms chunks
            data = wf.readframes(chunk_size)
            
            # Simple generator
            def request_generator():
                nonlocal data
                try:
                    if len(data) == 0: return
                    # config is passed to streaming_recognize, so don't yield it here
                    
                    while len(data) > 0:
                        yield speech.StreamingRecognizeRequest(audio_content=data)
                        time.sleep(0.1) # Real-time simulation
                        data = wf.readframes(chunk_size)
                except Exception:
                    traceback.print_exc()

            requests = request_generator()
            start_time = time.time()
            
            try:
                responses = client.streaming_recognize(config=streaming_config, requests=requests)

                for response in responses:
                    if not response.results:
                        continue

                    result = response.results[0]
                    if not result.alternatives:
                        continue
                    
                    transcript = result.alternatives[0].transcript
                    elapsed = time.time() - start_time

                    if result.is_final:
                        print(f"[{elapsed:.3f}s] FINAL: '{transcript}'")
                        if single_utterance:
                            # single_utterance closes the stream after first result, checking if that happens
                            print("Single Utterance finalized. Stream closed by server.")
                            break 
                    else:
                        if log_interim_stability:
                            stability = result.stability
                            print(f"[{elapsed:.3f}s] Interim | Stability: {stability:.2f} | Transcript: '{transcript}'")

            except Exception as e:
                # single_utterance might throw an error when stream closes?
                # or just StopIteration. 
                print(f"Stream ended or error: {e}")


def run_pivot_suite():
    # Audio files to test (Phase 1 files)
    files = ["test_200ms.wav", "test_400ms.wav", "test_600ms.wav", "test_1000ms.wav"]
    # Verify files exist
    files = [f for f in files if os.path.exists(f)]
    if not files:
        print("No audio files found!")
        return

    # Test A: Standard Model
    run_stt_test(
        "Test A: Standard Model (default)", 
        files, 
        {"model": "default"}
    )

    # Test B: Single Utterance
    run_stt_test(
        "Test B: Single Utterance Mode", 
        files, 
        {"model": "default", "single_utterance": True}
    )

    # Test C: Interim Stability
    # We only need to run this on one interesting file (e.g. 1000ms) to reduce noise, or all.
    # User asked for "Run the test again...". Let's run on 1000ms to see if we spot the pause stability.
    run_stt_test(
        "Test C: Interim Stability (1000ms only)", 
        ["test_1000ms.wav"], 
        {"model": "default", "log_interim_stability": True}
    )

if __name__ == "__main__":
    run_pivot_suite()
