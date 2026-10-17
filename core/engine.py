"""Central transcription orchestrator coordinating audio capture, STT, and translation."""

import asyncio
import os
import time
from typing import Any, Callable, Dict, List, Optional

from core.audio import AudioStream
from core.stt.cloud_engine import CloudSTTEngine
from core.stt.local_engine import LocalSTTEngine
from core.stt.runpod_engine import RunPodSTTEngine
from core.translation import Translator
from core.utils import parse_bool, resolve_pipeline_mode, resource_path


class TranscriptionEngine:
    """Manages audio capture, speech recognition, and multilingual translations."""

    def __init__(self, broadcast_callback: Callable[[Dict[str, Any]], Any]):
        self.pipeline_mode = resolve_pipeline_mode()

        # State (shared across all engines)
        self.source_lang = os.getenv("SOURCE_LANG", "nl-NL")
        self.target_lang_1 = os.getenv("TARGET_LANG_1", "en")
        self.target_lang_2 = os.getenv("TARGET_LANG_2", "ru")
        self.target_lang_3 = os.getenv("TARGET_LANG_3", "de")
        self.target_lang_4 = os.getenv("TARGET_LANG_4", "nl-NL")
        self.enable_trans_3 = parse_bool(os.getenv("ENABLE_TRANS_3", "false"))
        self.enable_trans_4 = parse_bool(os.getenv("ENABLE_TRANS_4", "false"))
        self.is_paused = False
        self.restart_required = False

        self.broadcast = broadcast_callback
        self.audio_stream = AudioStream()

        # Timing / VAD state
        self.silence_frames = 0
        self.speech_frames = 0
        self.current_seg_id = f"seg_{int(time.time()*1000)}"
        self.last_translate_time = 0.0
        self.translate_interval = int(os.getenv("TRANSLATE_INTERVAL_SEC", "10"))
        self.silence_threshold_ms = int(os.getenv("SILENCE_THRESHOLD_MS", "300"))
        self.translate_count = 0
        self.session_start_time = time.time()
        self.MAX_DURATION_SECONDS = int(os.getenv("MAX_SESSION_MINUTES", "30")) * 60
        self.min_local_chunk_s = float(os.getenv("MIN_LOCAL_CHUNK_S", "5.0"))
        self.interim_chunk_s = float(os.getenv("INTERIM_CHUNK_S", "7.0"))
        self.max_local_chunk_s = float(os.getenv("MAX_LOCAL_CHUNK_S", "25.0"))
        self.max_silence_flush_ms = int(os.getenv("MAX_SILENCE_FLUSH_MS", "2000"))
        self.min_speech_frames = 10
        self.last_transcript = ""

        # Load glossary if exists
        self.glossary_text = ""
        self.glossary_phrases: List[str] = []
        glos_path = resource_path("SpGlos.txt")
        if os.path.exists(glos_path):
            try:
                with open(glos_path, "r", encoding="utf-8") as f:
                    self.glossary_text = f.read().strip()
                    self.glossary_phrases = [p.strip() for p in self.glossary_text.split(",") if p.strip()]
                    print(f"Loaded glossary from {glos_path}: {len(self.glossary_phrases)} phrases")
            except Exception as e:
                print(f"Notice: Could not read glossary file {glos_path}: {e}")

        # Translation module
        self.translator = Translator(
            pipeline_mode=self.pipeline_mode,
            source_lang=self.source_lang,
        )

        # Translation tasks tracking (for cancellation of superseded in-flight translations)
        self.active_translation_tasks: Dict[str, asyncio.Task] = {}

        # Initialize appropriate STT engine
        if self.pipeline_mode == "cloud":
            self.stt_engine = CloudSTTEngine(self)
        elif self.pipeline_mode == "runpod":
            self.stt_engine = RunPodSTTEngine(self)
        else:
            self.stt_engine = LocalSTTEngine(self)

    # --- Backward-compatible helpers & config updates ---

    def _setup_recognition_config(self) -> None:
        if hasattr(self.stt_engine, "setup_recognition_config"):
            self.stt_engine.setup_recognition_config()

    def get_input_devices(self) -> List[Dict[str, Any]]:
        return self.audio_stream.list_devices()

    def update_config(self, config: Dict[str, Any]) -> None:
        print(f"Updating Config: {config}")

        if "paused" in config:
            was_paused = self.is_paused
            self.is_paused = config["paused"]
            self.audio_stream.playback_paused = self.is_paused
            if was_paused and not self.is_paused:
                self.session_start_time = time.time()

        if "source_lang" in config and config["source_lang"] != self.source_lang:
            self.source_lang = config["source_lang"]
            self.translator.source_lang = self.source_lang
            if self.pipeline_mode == "cloud":
                self._setup_recognition_config()
            self.restart_required = True

        if "target_lang_1" in config:
            self.target_lang_1 = config["target_lang_1"]

        if "target_lang_2" in config:
            self.target_lang_2 = config["target_lang_2"]

        if "target_lang_3" in config:
            self.target_lang_3 = config["target_lang_3"]

        if "target_lang_4" in config:
            self.target_lang_4 = config["target_lang_4"]

        if "enable_trans_3" in config:
            self.enable_trans_3 = config["enable_trans_3"]

        if "enable_trans_4" in config:
            self.enable_trans_4 = config["enable_trans_4"]

        if "device_index" in config:
            idx = config["device_index"]

            if isinstance(idx, str) and idx.startswith("file_"):
                file_name = idx.replace("file_", "", 1)
                os.environ["DEMO_FILE_PATH"] = file_name
                self.audio_stream.change_device("demo_file")
                self.restart_required = True
            elif idx == "demo_file":
                if self.audio_stream.current_device_index != "demo_file":
                    self.audio_stream.change_device("demo_file")
                    self.restart_required = True
            elif idx == "vban":
                if self.audio_stream.current_device_index != "vban":
                    self.audio_stream.change_device("vban")
                    self.restart_required = True
            elif isinstance(idx, str) and idx.startswith("sd_"):
                if self.audio_stream.current_device_index != idx:
                    self.audio_stream.change_device(idx)
                    self.restart_required = True
            elif idx == "default" or idx is None:
                try:
                    default = self.audio_stream.p.get_default_input_device_info()
                    self.audio_stream.change_device(int(default['index']))
                except Exception:
                    self.audio_stream.change_device(None)
                self.restart_required = True
            else:
                try:
                    self.audio_stream.change_device(int(idx))
                    self.restart_required = True
                except (ValueError, TypeError):
                    self.audio_stream.change_device(idx)
                    self.restart_required = True

    # --- Translation Dispatch ---

    async def translate_text_async(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate text using configured translation backend."""
        return await self.translator.translate_text_async(text, target_lang, source_lang=source_lang)

    async def handle_translation(
        self,
        seg_id: str,
        text: str,
        target_lang: str,
        col_key: str,
        chain_to: Optional[List[tuple]] = None,
        source_override: Optional[str] = None,
        is_interim: bool = False,
    ) -> None:
        """Background task to translate text and broadcast to UI."""
        if not text:
            return

        # Cancel any prior in-flight translation task for this column
        prev_task = self.active_translation_tasks.get(col_key)
        if prev_task and not prev_task.done():
            prev_task.cancel()

        curr_task = asyncio.current_task()
        if curr_task:
            self.active_translation_tasks[col_key] = curr_task

        try:
            translated = await self.translate_text_async(text, target_lang, source_lang=source_override)
            msg = {
                "action": "update_translation",
                "id": seg_id,
                "col_key": col_key,
                "text": translated,
            }
            await self.broadcast(msg)

            # Restrict chain_to back-translations to final chunks only
            if chain_to and not is_interim:
                for c_lang, c_key in chain_to:
                    asyncio.create_task(
                        self.handle_translation(
                            seg_id,
                            translated,
                            c_lang,
                            c_key,
                            source_override=target_lang,
                            is_interim=False,
                        )
                    )
        except asyncio.CancelledError:
            pass
        finally:
            if curr_task and self.active_translation_tasks.get(col_key) is curr_task:
                self.active_translation_tasks.pop(col_key, None)

    # --- Run Loop ---

    async def run(self) -> None:
        """Main loop — activates audio stream and executes selected STT engine."""
        print("Starting Audio Stream...")
        self.audio_stream.start()
        await self.stt_engine.run()
