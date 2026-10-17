"""Translation services for local (Ollama) and cloud (Google Cloud Translate) pipelines."""

import asyncio
import os
from typing import Optional

from core.utils import resource_path

LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "ru": "Russian",
    "uk": "Ukrainian", "fa": "Farsi", "ar": "Arabic", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "pl": "Polish",
}


class Translator:
    """Manages translations across Cloud (Google) and Local (Ollama) modes."""

    def __init__(
        self,
        pipeline_mode: str = "local",
        source_lang: str = "nl-NL",
        ollama_model: Optional[str] = None,
        ollama_url: Optional[str] = None,
        credentials_path: Optional[str] = None,
        project_id: Optional[str] = None,
    ):
        self.pipeline_mode = pipeline_mode.lower()
        self.source_lang = source_lang
        self.ollama_model = ollama_model or os.getenv("OLLAMA_MODEL", "qwen2.5:3b")
        self.ollama_url = ollama_url or os.getenv("OLLAMA_URL", "http://localhost:11434")

        self.translate_client = None
        self.parent = None

        if self.pipeline_mode == "cloud":
            creds = credentials_path or resource_path(os.getenv("GOOGLE_APPLICATION_CREDENTIALS", ""))
            proj = project_id or os.getenv("GOOGLE_PROJECT_ID")
            if creds and os.path.exists(creds) and proj:
                try:
                    from google.cloud import translate
                    self.translate_client = translate.TranslationServiceClient.from_service_account_json(creds)
                    self.parent = f"projects/{proj}/locations/global"
                except ImportError:
                    print("Warning: google-cloud-translate not installed.")

    async def translate_text_async(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Translate text using cloud or local backend."""
        if not text or "***" in text:
            return text
        if not target_lang:
            return ""

        source_code = source_lang.split("-")[0] if source_lang else self.source_lang.split("-")[0]
        if target_lang == source_code:
            return text

        if self.pipeline_mode == "cloud":
            return await self._translate_cloud(text, target_lang, source_lang)
        else:
            return await self._translate_local(text, target_lang, source_lang)

    async def _translate_cloud(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Google Cloud Translate."""
        if not self.translate_client or not self.parent:
            return text

        loop = asyncio.get_running_loop()
        source_code = source_lang.split("-")[0] if source_lang else self.source_lang.split("-")[0]

        def _call():
            try:
                response = self.translate_client.translate_text(
                    request={
                        "parent": self.parent,
                        "contents": [text],
                        "mime_type": "text/plain",
                        "source_language_code": source_code,
                        "target_language_code": target_lang,
                    }
                )
                return response.translations[0].translated_text
            except Exception as e:
                print(f"Translation Error ({target_lang}): {e}")
                return text

        return await loop.run_in_executor(None, _call)

    async def _translate_local(self, text: str, target_lang: str, source_lang: Optional[str] = None) -> str:
        """Ollama AsyncClient translation."""
        lang_name = LANG_NAMES.get(target_lang.split("-")[0], target_lang)
        source_code = source_lang.split("-")[0] if source_lang else self.source_lang.split("-")[0]
        source_lang_name = LANG_NAMES.get(source_code, source_code)

        system_prompt = (
            f"You are a professional, native translator. "
            f"Translate the provided {source_lang_name} text into {lang_name}. "
            f"Retain the original tone and meaning perfectly. "
            f"Provide ONLY the direct translation. "
            f"Do not provide explanations, introductions, quotes, or markdown."
        )

        try:
            from ollama import AsyncClient
            client = AsyncClient(host=self.ollama_url)
            response = await client.chat(
                model=self.ollama_model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                options={"temperature": 0},
            )
            return response["message"]["content"].strip()
        except Exception as e:
            print(f"Translation Error ({target_lang}): {e}")
            return text
