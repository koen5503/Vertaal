"""Local Speech-to-Text engine using Apple Silicon MLX-Whisper and VAD accumulation."""

import asyncio
import difflib
import os
import re
import string
import time
from typing import List, Optional, Set

try:
    import numpy as np
except ImportError:
    np = None

from core.audio import CHUNK_SIZE, FRAME_DURATION_MS, SAMPLE_RATE
from core.stt.base import BaseSTTEngine

try:
    import mlx_whisper
except ImportError:
    mlx_whisper = None


# ============================================================================
# Anti-Hallucination Filter
# ============================================================================
NOISE_HALLUCINATION_PHRASES: Set[str] = {
    # Dutch polite closings / noise hallucinations
    "dank u wel",
    "dank u",
    "dank je wel",
    "dank je",
    "hartelijk dank",
    "bedankt",
    "bedankt voor het kijken",
    "bedankt voor het luisteren",
    "tot ziens",
    "fijne dag",
    "fijne avond",
    "dag",
    # English polite closings / noise hallucinations
    "thank you",
    "thank you very much",
    "thanks",
    "thanks for watching",
    "thanks for listening",
    "bye",
    "goodbye",
    # TV / Subtitle credits
    "subtitles by",
    "captioned by",
    "amaraorg",
    "opensubtitles",
    "tv gelderland",
    "omroep gelderland",
    "rtv oost",
    "rtv noord",
    "omroep brabant",
}

BROADCAST_CREDIT_PATTERN = re.compile(
    r"^(?:tv|omroep|rtv)\s+[a-z]+(?:\s+\d{4})?$|"
    r"^(?:npo|rtl|nos)(?:\s+\d+)?(?:\s+\d{4})?$|"
    r"^(?:ondertiteling|ondertitels|ondertiteld(?:\s+door)?|subtitles(?:\s+by)?|captioned(?:\s+by)?).*$",
    re.IGNORECASE
)

TAIL_HALLUCINATION_PATTERN = re.compile(
    r"(?:\s*[\-\:\;]\s*|(?<=[\.\,\!\?])\s+|\s+|^)"
    r"(?:"
    r"(?:tv|omroep|rtv)\s+[a-z]+(?:\s+\d{4})?|"
    r"(?:npo|rtl|nos)(?:\s+\d+)?(?:\s+\d{4})?|"
    r"ondertiteling(?:\s+door\s+[a-z]+)?|"
    r"ondertitels|"
    r"bedankt voor het kijken|"
    r"bedankt voor het luisteren|"
    r"thanks for watching|"
    r"thanks for listening|"
    r"subtitles by\s+[a-z0-9\.]+|"
    r"captioned by\s+[a-z0-9\.]+"
    r")[\.\,\!\?\s]*$",
    re.IGNORECASE
)


def filter_anti_hallucination(
    text: Optional[str],
    last_transcript: str = "",
    glossary_phrases: Optional[List[str]] = None,
) -> Optional[str]:
    """Run anti-hallucination filters on Whisper transcription output."""
    if not text:
        return None

    # Strip trailing hallucination artifacts (broadcaster tags, polite closing tacks)
    cleaned_tail = TAIL_HALLUCINATION_PATTERN.sub("", text).strip()
    if not cleaned_tail:
        print(f"Filtered hallucination (pure broadcast/credit): '{text}'")
        return None
    if cleaned_tail != text.strip():
        print(f"Trimmed trailing hallucination credit from text: '{text}' -> '{cleaned_tail}'")
        text = cleaned_tail

    clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
    clean_text = " ".join(clean_text.split())
    words = clean_text.split()

    if not words:
        return None

    # 0. Noise / Polite Closing / Broadcaster Hallucination Filter
    clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
    if (
        clean_text in NOISE_HALLUCINATION_PHRASES
        or clean_no_digits in NOISE_HALLUCINATION_PHRASES
        or BROADCAST_CREDIT_PATTERN.match(clean_text)
        or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
    ):
        print(f"Filtered hallucination (noise/broadcaster credit): '{text}'")
        return None

    # Prompt Leakage Filter (Whisper Hallucination)
    last_trans = last_transcript or ""
    if last_trans:
        clean_last = last_trans.translate(str.maketrans('', '', string.punctuation)).lower()
        last_words = clean_last.split()

        # a) Exact Prompt Leakage
        if clean_text == clean_last:
            print(f"Filtered hallucination (Exact Prompt Leakage): '{text}'")
            return None

        # b) Partial Prompt Leakage (len >= 3 and clean_text in clean_last)
        if len(words) >= 3 and clean_text in clean_last:
            print(f"Filtered hallucination (Partial Prompt Leakage): '{text}'")
            return None

        # c) Overlapping sliding-window / sub-sequence prompt-leak trimmer using SequenceMatcher
        matcher = difflib.SequenceMatcher(None, last_words, words)
        match = matcher.find_longest_match(0, len(last_words), 0, len(words))

        is_large_leak = (match.size >= 4 and match.b <= 5) or (match.size >= 6 and match.b <= 15)
        is_tail_overlap = (match.size >= 1 and match.b == 0 and match.a + match.size == len(last_words))
        is_tail_sub = (match.size >= 2 and match.b <= 5 and match.a + match.size == len(last_words))

        if is_large_leak or is_tail_overlap or is_tail_sub:
            orig_words = text.split()
            cut_idx = match.b + match.size
            trimmed_words = orig_words[cut_idx:]
            text = " ".join(trimmed_words).strip()
            if not text:
                print(f"Filtered hallucination (100% Leaked Prompt Drop): matched {match.size} words")
                return None
            print(f"Trimmed {match.size} leaked prompt words (matched at idx {match.b}). Kept: '{text}'")

            clean_text = text.translate(str.maketrans('', '', string.punctuation)).lower().strip()
            clean_text = " ".join(clean_text.split())
            words = clean_text.split()
            if not words:
                return None

            clean_no_digits = " ".join(re.sub(r"\b(19\d\d|20\d\d)\b", "", clean_text).split())
            if (
                clean_text in NOISE_HALLUCINATION_PHRASES
                or clean_no_digits in NOISE_HALLUCINATION_PHRASES
                or BROADCAST_CREDIT_PATTERN.match(clean_text)
                or BROADCAST_CREDIT_PATTERN.match(clean_no_digits)
            ):
                print(f"Filtered hallucination (noise/broadcaster credit after trim): '{text}'")
                return None

    # 1. Repetition filter
    # A. Exactly the same word repeated
    if len(words) >= 2 and len(set(words)) == 1:
        print(f"Filtered hallucination (pure repetitive): '{text}'")
        return None

    # B. Consecutive repeating words OR extremely low unique word count
    max_consecutive = 1
    current_consecutive = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            current_consecutive += 1
            max_consecutive = max(max_consecutive, current_consecutive)
        else:
            current_consecutive = 1

    if max_consecutive >= 4:
        print(f"Filtered hallucination (consecutive word loop {max_consecutive}x): '{text}'")
        return None

    # E.g. 10 words, but only 2 unique (< 30%)
    if len(words) >= 5 and len(set(words)) / len(words) < 0.3:
        print(f"Filtered hallucination (low unique ratio): '{text}'")
        return None

    # 2. Isolated glossary word filter
    # When VAD catches a breath/noise, Whisper defaults to outputting 1-2 words from initial_prompt
    if glossary_phrases and len(words) <= 2:
        glos_lower = [p.lower() for p in glossary_phrases]
        if any(w in glos_lower for w in words) or clean_text in glos_lower:
            print(f"Filtered hallucination (isolated glossary term): '{text}'")
            return None

    return text if text else None


# ============================================================================
# LocalSTTEngine
# ============================================================================
class LocalSTTEngine(BaseSTTEngine):
    """Local STT Engine using Apple Silicon MLX-Whisper."""

    def __init__(self, engine):
        self.engine = engine
        self.whisper_model = os.getenv("WHISPER_MODEL", "mlx-community/whisper-small-mlx")
        self.min_speech_frames = 10
        self.audio_buffer = bytearray()
        self.silence_frames = 0
        self.speech_frames = 0
        self.last_transcript = ""
        self.active_seg_id: Optional[str] = None
        self.last_interim_s: float = 0.0

    def transcribe_buffer(self) -> Optional[str]:
        """Convert accumulated audio buffer to numpy and run mlx-whisper."""
        if len(self.audio_buffer) < CHUNK_SIZE * 2:
            return None

        if not mlx_whisper:
            print("Error: mlx_whisper module not available on this system.")
            return None

        audio_np = np.frombuffer(bytes(self.audio_buffer), dtype=np.int16).astype(np.float32) / 32768.0
        whisper_lang = self.engine.source_lang.split("-")[0]

        transcribe_kwargs = {
            "path_or_hf_repo": self.whisper_model,
            "language": whisper_lang,
        }

        prompt_parts = []
        if self.engine.glossary_text:
            prompt_parts.append(self.engine.glossary_text)

        last_trans = self.last_transcript or getattr(self.engine, "last_transcript", "")
        if last_trans:
            prompt_parts.append(last_trans)

        if prompt_parts:
            transcribe_kwargs["initial_prompt"] = " ".join(prompt_parts).strip()

        try:
            result = mlx_whisper.transcribe(
                audio_np,
                **transcribe_kwargs
            )
            raw_text = result.get("text", "").strip()
            return filter_anti_hallucination(
                raw_text,
                last_transcript=self.last_transcript,
                glossary_phrases=self.engine.glossary_phrases,
            )
        except Exception as e:
            print(f"Whisper Transcription Error: {e}")
            return None

    async def run(self) -> None:
        """Accumulate audio and transcribe on silence via MLX-Whisper."""
        loop = asyncio.get_event_loop()
        chunk_count = 0

        while True:
            # Handle restart FIRST (device/file switch sets running=False)
            if self.engine.restart_required:
                print("Restarting stream due to config change...")
                self.engine.restart_required = False
                self.audio_buffer = bytearray()
                self.silence_frames = 0
                self.speech_frames = 0
                self.active_seg_id = None
                self.last_interim_s = 0.0
                self.engine.current_seg_id = f"seg_{int(time.time()*1000)}"
                if not self.engine.audio_stream.running:
                    self.engine.audio_stream.start()
                continue

            if not self.engine.audio_stream.running:
                await asyncio.sleep(0.5)
                continue

            if self.engine.is_paused:
                await asyncio.sleep(0.1)
                continue

            if time.time() > self.engine.session_expiry_time:
                print(f"TIMEOUT REACHED (expiry={self.engine.session_expiry_time:.1f}). Auto-pausing.")
                self.engine.is_paused = True
                await self.engine.broadcast({"action": "auto_paused", "reason": f"{int(self.engine.initial_session_window_minutes)}_min_limit"})
                continue

            chunk = await loop.run_in_executor(None, self.engine.audio_stream.read_chunk)
            if not chunk:
                continue

            chunk_count += 1

            if chunk_count % 10 == 0:
                rms = self.engine.audio_stream.calculate_rms(chunk)
                await self.engine.broadcast({"type": "volume", "rms": rms})

            is_speech = self.engine.audio_stream.is_speech(chunk)
            if is_speech:
                self.silence_frames = 0
                self.speech_frames += 1
            else:
                self.silence_frames += 1

            self.audio_buffer.extend(chunk)

            # Keep max 1.0s pre-roll silence while waiting for speech
            if self.speech_frames == 0:
                PRE_ROLL_BYTES = 32000
                if len(self.audio_buffer) > PRE_ROLL_BYTES:
                    self.audio_buffer = self.audio_buffer[-PRE_ROLL_BYTES:]

            silence_ms = self.silence_frames * FRAME_DURATION_MS
            buffer_duration_s = len(self.audio_buffer) / (SAMPLE_RATE * 2)

            is_natural_pause = (silence_ms >= self.engine.silence_threshold_ms and buffer_duration_s >= self.engine.min_local_chunk_s)
            is_force_flush = (silence_ms >= self.engine.max_silence_flush_ms)
            is_hard_ceiling = (buffer_duration_s >= self.engine.max_local_chunk_s)
            is_interim_interval = (
                (buffer_duration_s - self.last_interim_s >= self.engine.interim_chunk_s)
                and not is_natural_pause
                and not is_hard_ceiling
            )

            # --- Finalization Trigger (Natural Pause, Max Silence Flush, or Hard Ceiling) ---
            if (is_natural_pause or is_force_flush or is_hard_ceiling) and self.speech_frames > 0:
                # Discard trapped pure mic clicks (under 90ms)
                if is_force_flush and self.speech_frames < 3:
                    self.audio_buffer = bytearray()
                    self.speech_frames = 0
                    self.silence_frames = 0
                    self.active_seg_id = None
                    self.last_interim_s = 0.0
                    continue

                met_speech_req = self.speech_frames >= self.min_speech_frames or ((is_force_flush or is_hard_ceiling) and self.speech_frames >= 3)
                if not met_speech_req:
                    continue

                seg_id = self.active_seg_id or f"seg_{int(time.time()*1000)}"
                reason = "hard_ceiling" if is_hard_ceiling else ("force_flush" if is_force_flush else f"pause_{silence_ms}ms")
                print(f"Finalizing {seg_id} ({reason}, dur={buffer_duration_s:.1f}s, speech={self.speech_frames}f)...")

                transcript = await loop.run_in_executor(None, self.transcribe_buffer)

                if transcript:
                    print(f"STT [final]: '{transcript}'")
                    self.engine.extend_session_window(reason="stt_activity")
                    self.last_transcript = transcript[-1200:]
                    self.engine.last_transcript = self.last_transcript

                    msg = {
                        "id": seg_id,
                        "status": "final",
                        "col1_text": transcript,
                    }
                    await self.engine.broadcast(msg)

                    self.engine.translate_count += len(transcript) * 2
                    log_text = f"'{transcript[:50]}'..." if len(transcript) > 50 else f"'{transcript}'"
                    print(f"Translate ({self.engine.translate_count} chars total): {log_text}")

                    chain_3 = [(self.engine.target_lang_3, "trans3")] if getattr(self.engine, "enable_trans_3", True) else []
                    asyncio.create_task(
                        self.engine.handle_translation(
                            seg_id,
                            transcript,
                            self.engine.target_lang_1,
                            "trans1",
                            chain_to=chain_3,
                            is_interim=False
                        )
                    )
                    chain_4 = [(self.engine.target_lang_4, "trans4")] if getattr(self.engine, "enable_trans_4", True) else []
                    asyncio.create_task(
                        self.engine.handle_translation(
                            seg_id,
                            transcript,
                            self.engine.target_lang_2,
                            "trans2",
                            chain_to=chain_4,
                            is_interim=False
                        )
                    )

                self.audio_buffer = bytearray()
                self.silence_frames = 0
                self.speech_frames = 0
                self.active_seg_id = None
                self.last_interim_s = 0.0
                self.engine.current_seg_id = f"seg_{int(time.time()*1000)}"

            # --- Incremental Interim Trigger (Option B: continuous speech reaching interim interval) ---
            elif is_interim_interval and self.speech_frames >= getattr(self.engine, 'min_speech_frames', 10):
                if not self.active_seg_id:
                    self.active_seg_id = f"seg_{int(time.time()*1000)}"
                seg_id = self.active_seg_id

                print(f"Interim update for {seg_id} (dur={buffer_duration_s:.1f}s, speech={self.speech_frames}f)...")
                transcript = await loop.run_in_executor(None, self.transcribe_buffer)

                if transcript:
                    print(f"STT [interim]: '{transcript}'")
                    self.engine.extend_session_window(reason="stt_activity")
                    msg = {
                        "id": seg_id,
                        "status": "interim",
                        "col1_text": transcript,
                    }
                    await self.engine.broadcast(msg)

                    chain_3 = [(self.engine.target_lang_3, "trans3")] if getattr(self.engine, "enable_trans_3", True) else []
                    asyncio.create_task(
                        self.engine.handle_translation(
                            seg_id,
                            transcript,
                            self.engine.target_lang_1,
                            "trans1",
                            chain_to=chain_3,
                            is_interim=True
                        )
                    )
                    chain_4 = [(self.engine.target_lang_4, "trans4")] if getattr(self.engine, "enable_trans_4", True) else []
                    asyncio.create_task(
                        self.engine.handle_translation(
                            seg_id,
                            transcript,
                            self.engine.target_lang_2,
                            "trans2",
                            chain_to=chain_4,
                            is_interim=True
                        )
                    )

                self.last_interim_s = buffer_duration_s
