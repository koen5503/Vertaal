import os
import wave
from google.cloud import texttospeech

def generate_speech(text, output_filename):
    """Generates speech for a given text using Google Cloud TTS."""
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)

    # Note: We use nl-NL-Standard-A for a generic Dutch voice.
    voice = texttospeech.VoiceSelectionParams(
        language_code="nl-NL",
        name="nl-NL-Standard-A",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )

    # Select the type of audio file you want returned
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000
    )

    try:
        response = client.synthesize_speech(
            input=input_text, voice=voice, audio_config=audio_config
        )
        # The response's audio_content is binary.
        with open(output_filename, "wb") as out:
            out.write(response.audio_content)
        print(f"Audio content written to file {output_filename}")
    except Exception as e:
        print(f"Google TTS failed: {e}. Falling back to macOS 'say' command.")
        # Fallback to macOS 'say' command
        # say -o output.aiff --data-format=LEI16@16000 "text"
        # However, say usually outputs aiff or m4a. We need wav.
        # We can output to aiff then convert or just try to write to wav if supported (macOS say sometimes supports --file-format=WAVE)
        
        # Simpler: say -> aiff -> ffmpeg/sox? No, we want pure python/standard tools if possible. 
        # Actually `say` on modern macos supports --data-format. 
        # Let's try to generate a .wav directly.
        # "say -o output.wav --data-format=LEI16@16000" might work.
        
        # Safe fallback: create aiff then direct read? 
        # Let's try simplified: say "text" -o temp.aiff
        # Then convert. But we need 16k mono.
        
        # Let's try to just use os.system with a highly specific command
        # -o {output_filename} --data-format=LEI16@16000 is ideal but might not correspond to WAV header.
        # `say` often creates AIFF-C for LEI16.
        
        # Workaround: Use 'say' to generate an AIFF file, then Convert to WAV using similar logic or just check if we can write raw samples.
        # Since we are using `wave` module, we need a valid WAV file.
        # Let's assume the user has `ffmpeg` or `sox`? No, better not assume.
        # ACTUALLY: The user asked for "Google Cloud TTS". 
        # But for *latency testing*, any audio works.
        
        # Let's try:
        # say -o {output_filename} --file-format=WAVE --data-format=LEI16@16000 "text"
        ret = os.system(f'say -v Xander -o "{output_filename}" --file-format=WAVE --data-format=LEI16@16000 "{text}"')
        if ret != 0:
             # Fallback to default voice if Xander (NL) missing
             ret = os.system(f'say -o "{output_filename}" --file-format=WAVE --data-format=LEI16@16000 "{text}"')
        
        if ret == 0:
            print(f"Generated {output_filename} using macOS 'say'")
        else:
            raise RuntimeError("Both Google TTS and macOS 'say' fallback failed.")

def create_silence(duration_ms, sample_rate=16000, channels=1, sample_width=2):
    """Creates a bytes object representing silence."""
    num_samples = int(sample_rate * (duration_ms / 1000.0))
    # 16-bit PCM = 2 bytes per sample
    # silence is all zeros
    return b'\x00' * (num_samples * channels * sample_width)

def concatenate_wavs(input_files, output_filename, silence_ms):
    """Concatenates wav files with silence in between."""
    data = []
    
    # Read first file to get params
    with wave.open(input_files[0], 'rb') as w:
        params = w.getparams()
        # params: (nchannels, sampwidth, framerate, nframes, comptype, compname)
    
    silence_data = create_silence(silence_ms, params.framerate, params.nchannels, params.sampwidth)

    for i, file in enumerate(input_files):
        with wave.open(file, 'rb') as w:
            current_params = w.getparams()
            # Compare only nchannels, sampwidth, framerate
            if (current_params.nchannels != params.nchannels or 
                current_params.sampwidth != params.sampwidth or 
                current_params.framerate != params.framerate):
                raise ValueError(f"File {file} has different parameters than the first file.")
            
            data.append(w.readframes(w.getnframes()))
        
        # Add silence if not the last file
        if i < len(input_files) - 1:
            data.append(silence_data)
    
    with wave.open(output_filename, 'wb') as w:
        w.setparams(params)
        for d in data:
            w.writeframes(d)
    print(f"Exported {output_filename}")

def generate_test_set():
    sentences = [
        "Laten wij het boek openen",
        "Het is belangrijk om te zien",
        "Wij moeten bidden voor wijsheid",
        "De tekst spreekt tot ons hart",
        "Laten we samen luisteren naar het woord"
    ]

    # Generate individual sentence files
    filenames = []
    print("Generating sentence audio files...")
    for i, sentence in enumerate(sentences):
        filename = f"sentence_{i}.wav"
        generate_speech(sentence, filename)
        filenames.append(filename)
    
    silence_durations_ms = [200, 400, 600, 1000, 2000, 3000]

    for silence_ms in silence_durations_ms:
        print(f"Creating merged audio with {silence_ms}ms silence...")
        output_filename = f"test_{silence_ms}ms.wav"
        concatenate_wavs(filenames, output_filename, silence_ms)

    # Cleanup individual files
    for f in filenames:
        try:
             os.remove(f)
        except OSError:
            pass

if __name__ == "__main__":
    # Ensure google credentials are set
    if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
       print("Warning: GOOGLE_APPLICATION_CREDENTIALS environment variable is not set.")
    
    try:
        generate_test_set()
    except Exception as e:
        print(f"An error occurred: {e}")
