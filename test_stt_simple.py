from google.cloud import speech
import io

def run_quickstart():
    # Instantiates a client
    client = speech.SpeechClient.from_service_account_json(
        "/Users/koen/Ondertitels/ondertitels-486017-0ee48ab1ba8d.json"
    )

    # The name of the audio file to transcribe
    file_name = "test_200ms.wav"

    # Loads the audio into memory
    with io.open(file_name, "rb") as audio_file:
        content = audio_file.read()

    audio = speech.RecognitionAudio(content=content)

    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="nl-NL",
    )

    # Detects speech in the audio file
    print("Sending synchronous request...")
    try:
        response = client.recognize(config=config, audio=audio)

        for result in response.results:
            print("Transcript: {}".format(result.alternatives[0].transcript))
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    run_quickstart()
