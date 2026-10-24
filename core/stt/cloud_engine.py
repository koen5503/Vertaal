"""Cloud Speech-to-Text engine using Google Cloud Speech streaming API."""

import asyncio
import os
import threading
import time
from typing import Optional

from core.audio import FRAME_DURATION_MS, SAMPLE_RATE
from core.stt.base import BaseSTTEngine
from core.utils import resource_path

try:
    from google.cloud import speech
except ImportError:
    speech = None


class CloudSTTEngine(BaseSTTEngine):
    """Google Cloud Speech streaming STT engine."""

    def __init__(self, engine):
        self.engine = engine
        self.credentials_path = resource_path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
        self.project_id = os.getenv("GOOGLE_PROJECT_ID")

        if not self.credentials_path or not os.path.exists(self.credentials_path):
            raise ValueError("GOOGLE_APPLICATION_CREDENTIALS is not set or file does not exist. Check your .env file.")
        if not self.project_id:
            raise ValueError("GOOGLE_PROJECT_ID is not set. Check your .env file.")

        if speech is None:
            raise ImportError("google-cloud-speech is required for cloud mode.")

        self.speech_client = speech.SpeechClient.from_service_account_json(self.credentials_path)
        self.stability_threshold = float(os.getenv("STABILITY_THRESHOLD", "0.8"))
        self.min_words = int(os.getenv("MIN_WORDS_FOR_STABILITY", "5"))
        self.last_translated_text = ""
        self.config = None
        self.streaming_config = None
        self.setup_recognition_config()
        print("Cloud STT: Google Speech-to-Text initialized")

    def setup_recognition_config(self) -> None:
        """Construct Google Cloud RecognitionConfig and StreamingRecognitionConfig."""
        speech_contexts = []
        if self.engine.glossary_phrases:
            speech_contexts.append(speech.SpeechContext(phrases=self.engine.glossary_phrases))

        self.config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=SAMPLE_RATE,
            language_code=self.engine.source_lang,
            enable_automatic_punctuation=True,
            model="default",
            use_enhanced=True,
            speech_contexts=speech_contexts,
        )
        self.streaming_config = speech.StreamingRecognitionConfig(
            config=self.config,
            interim_results=True,
        )

    async def run(self) -> None:
        """Stream audio chunks to Google STT."""
        loop = asyncio.get_event_loop()

        while True:
            if not self.engine.audio_stream.running:
                await asyncio.sleep(0.5)
                continue

            if self.engine.is_paused:
                await asyncio.sleep(0.1)
                continue

            self.engine.restart_required = False
            stop_event = threading.Event()

            await loop.run_in_executor(None, self._run_google_stream_sync, stop_event, loop)

            if self.engine.restart_required:
                print("Restarting stream due to config change...")
                if not self.engine.audio_stream.running:
                    self.engine.audio_stream.start()

            self.engine.current_seg_id = f"seg_{int(time.time()*1000)}"

    def _run_google_stream_sync(self, stop_event: threading.Event, loop: asyncio.AbstractEventLoop) -> None:
        """Blocking function to handle one Google Stream session."""
        generator_stop = False

        def generator():
            nonlocal generator_stop
            chunk_count = 0

            if not self.engine.is_paused and chunk_count == 0:
                self.engine.reset_session_timer(reason="cloud_stream_start")

            while not generator_stop:
                if self.engine.restart_required or self.engine.is_paused:
                    generator_stop = True
                    return

                if time.time() > self.engine.session_expiry_time:
                    print(f"TIMEOUT REACHED (expiry={self.engine.session_expiry_time:.1f}). Auto-pausing.")
                    self.engine.is_paused = True
                    msg = {"action": "auto_paused", "reason": f"{int(self.engine.initial_session_window_minutes)}_min_limit"}
                    asyncio.run_coroutine_threadsafe(self.engine.broadcast(msg), loop=loop)
                    stop_event.set()
                    return

                chunk_count += 1
                chunk = self.engine.audio_stream.read_chunk()
                if not chunk:
                    generator_stop = True
                    break

                if chunk_count % 10 == 0:
                    rms = self.engine.audio_stream.calculate_rms(chunk)
                    vol_msg = {"type": "volume", "rms": rms}
                    asyncio.run_coroutine_threadsafe(self.engine.broadcast(vol_msg), loop=loop)

                is_speech = self.engine.audio_stream.is_speech(chunk)
                if is_speech:
                    self.engine.silence_frames = 0
                else:
                    self.engine.silence_frames += 1

                yield speech.StreamingRecognizeRequest(audio_content=chunk)

        try:
            responses = self.speech_client.streaming_recognize(
                config=self.streaming_config,
                requests=generator()
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

                print(f"STT: '{transcript}' | Words: {len(transcript.split())} | Stability: {stability:.2f} | Final: {is_final}")

                word_count = len(transcript.split())
                silence_ms = self.engine.silence_frames * FRAME_DURATION_MS

                eff_stability = 1.0 if word_count < self.min_words else self.stability_threshold
                sentence_end = transcript.rstrip().endswith(('.', '!', '?'))
                status = "final" if is_final else "interim"

                if (silence_ms > self.engine.silence_threshold_ms or sentence_end) and stability > eff_stability:
                    if not is_final:
                        reason = f"Punctuation '{transcript[-1]}'" if sentence_end else f"Silence {silence_ms}ms"
                        print(f"FORCE FINAL: '{transcript}' ({reason}) — waiting for Google final...")
                        generator_stop = True

                if is_final:
                    stop_event.set()

                msg = {
                    "id": self.engine.current_seg_id,
                    "status": status,
                    "col1_text": transcript,
                }
                self.engine.extend_session_window(reason="stt_activity")
                asyncio.run_coroutine_threadsafe(self.engine.broadcast(msg), loop=loop)

                now = time.time()
                if is_final or (stability > eff_stability and now - self.engine.last_translate_time >= self.engine.translate_interval):
                    self.engine.last_translate_time = now
                    self.engine.translate_count += len(transcript) * 2
                    print(f"Translate ({self.engine.translate_count} chars total): '{transcript[:50]}'..." if len(transcript) > 50 else f"Translate ({self.engine.translate_count} chars total): '{transcript}'")
                    chain_3 = [(self.engine.target_lang_3, "trans3")] if getattr(self.engine, "enable_trans_3", True) else []
                    asyncio.run_coroutine_threadsafe(
                        self.engine.handle_translation(
                            self.engine.current_seg_id,
                            transcript,
                            self.engine.target_lang_1,
                            "trans1",
                            chain_to=chain_3,
                            is_interim=not is_final
                        ),
                        loop=loop
                    )
                    chain_4 = [(self.engine.target_lang_4, "trans4")] if getattr(self.engine, "enable_trans_4", True) else []
                    asyncio.run_coroutine_threadsafe(
                        self.engine.handle_translation(
                            self.engine.current_seg_id,
                            transcript,
                            self.engine.target_lang_2,
                            "trans2",
                            chain_to=chain_4,
                            is_interim=not is_final
                        ),
                        loop=loop
                    )

                if stop_event.is_set():
                    break

        except Exception as e:
            print(f"Stream Error: {e}")
            stop_event.set()
