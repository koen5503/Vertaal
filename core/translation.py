"""Translation services for local (Ollama) and cloud (Google Cloud Translate) pipelines."""

import asyncio
import os
import re
from typing import Optional

from core.utils import resource_path

LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "ru": "Russian",
    "uk": "Ukrainian", "fa": "Farsi", "ar": "Arabic", "nl": "Dutch",
    "es": "Spanish", "pt": "Portuguese", "it": "Italian", "zh": "Chinese",
    "ja": "Japanese", "ko": "Korean", "tr": "Turkish", "pl": "Polish",
}

META_PATTERNS = [
    # English
    r'(?i)\bthe word\s+["\'«„\w]+.*?\btranslates\b.*',
    r'(?i)\bhowever,?\s+without\s+context\b.*',
    r'(?i)\bplease\s+provide\s+(?:the\s+)?(?:full|more|additional)\b.*',
    r'(?i)\bwithout\s+(?:more\s+)?context\b.*',
    r'(?i)\bas\s+per\s+your\s+strict\s+requirements\b.*',
    r'(?i)\bfor\s+example,?\s+if\s+the\s+full\s+sentence\b.*',
    # French
    r'(?i)\ble mot\s+[«"\'\w]+.*?\bse traduit\b.*',
    r'(?i)\bcependant,?\s+sans\s+contexte\b.*',
    r'(?i)\bsans\s+(?:plus\s+de\s+)?contexte\b.*',
    r'(?i)\bveuillez\s+fournir\b.*',
    r'(?i)\bselon\s+vos\s+exigences\s+strictes\b.*',
    r'(?i)\bpar\s+exemple,?\s+si\s+la\s+phrase\b.*',
    # Dutch
    r'(?i)\bhet woord\s+["\'\w]+.*?\bvertaalt\b.*',
    r'(?i)\bechter,?\s+zonder\s+context\b.*',
    r'(?i)\bzonder\s+(?:nadere\s+)?context\b.*',
    r'(?i)\bgeef\s+alstublieft\b.*',
    r'(?i)\bvolgens\s+uw\s+strikte\s+eisen\b.*',
    r'(?i)\bbijvoorbeeld,?\s+als\s+de\b.*',
    # German
    r'(?i)\bdas wort\s+["\'„\w]+.*?\bübersetzt\b.*',
    r'(?i)\bjedoch,?\s+ohne\s+kontext\b.*',
    r'(?i)\bohne\s+(?:weiteren\s+)?kontext\b.*',
    # Russian
    r'(?i)\bслово\s+[«"\'\w]+.*?\bпереводится\b.*',
    r'(?i)\bоднако,?\s+без\s+контекста\b.*',
    r'(?i)\bбез\s+(?:дополнительного\s+)?контекста\b.*',
    # Generic note markers across languages
    r'(?i)\b(?:note|remarque|hinweis|opmerking|примечание|explanation|explication|erklärung|uitleg)\s*:.*',
    r'(?i)\b(?:here\s+is\s+the\s+translation|voici\s+la\s+traduction|hier\s+is\s+de\s+vertaling|hier\s+ist\s+die\s+übersetzung|вот\s+перевод)\s*:?\s*',
]


def clean_translation_meta(text: str) -> str:
    """Strip conversational disclaimers, lecturing, and metadata from LLM translations."""
    if not text:
        return ""
    cleaned = text.strip()

    # If the LLM output a multi-line response with blank line, translation is paragraph 1
    if "\n\n" in cleaned:
        cleaned = cleaned.split("\n\n")[0].strip()

    # Strip introductory markers
    cleaned = re.sub(
        r'(?i)^(?:here\s+is\s+the\s+translation:?\s*|voici\s+la\s+traduction:?\s*|translation:?\s*|traduction:?\s*|vertaling:?\s*|translated\s+text:?\s*)',
        '',
        cleaned
    ).strip()

    # Find earliest trailing meta-commentary pattern
    earliest_idx = len(cleaned)
    for pat in META_PATTERNS:
        if "here" in pat or "voici" in pat or "hier" in pat:
            continue
        m = re.search(pat, cleaned)
        if m and m.start() < earliest_idx:
            earliest_idx = m.start()

    if earliest_idx < len(cleaned):
        cleaned = cleaned[:earliest_idx].strip()

    # Strip surrounding quotes if wrapped (including French « » and German „ “)
    if (cleaned.startswith('"') and cleaned.endswith('"')) or (cleaned.startswith("'") and cleaned.endswith("'")):
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith('«') and cleaned.endswith('»'):
        cleaned = cleaned[1:-1].strip()
    elif cleaned.startswith('„') and cleaned.endswith('“'):
        cleaned = cleaned[1:-1].strip()

    return cleaned


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
        lang_code = target_lang.split("-")[0].lower() if target_lang else "en"
        lang_name = LANG_NAMES.get(lang_code, target_lang)
        source_code = (source_lang.split("-")[0] if source_lang else self.source_lang.split("-")[0]).lower()
        source_lang_name = LANG_NAMES.get(source_code, source_code)

        system_prompt = (
            f"You are an automated real-time speech subtitle translation engine for church services.\n"
            f"Translate from {source_lang_name} to {lang_name}.\n\n"
            "CORE DIRECTIVES:\n"
            "1. You are a silent translation machine, not an assistant. You have no personality and never converse.\n"
            f"2. Output ONLY the raw translated text in {lang_name}. Absolutely zero explanations, notes, intros, or conversation.\n"
            "3. The input is live spoken audio transcribed in real-time. It frequently contains sentence fragments, pauses, and dangling words (e.g. 'over.', 'en.', 'maar.').\n"
            f"4. When input ends on an incomplete thought or trailing word, translate those exact words directly into {lang_name} and stop immediately.\n"
            "5. Never explain words, never comment on missing context or grammar, and never ask for more context in any language.\n"
            "6. Translate accurately and without omissions. Never omit proper names, biblical figures (e.g. David, Jesus, Moses, Paul), locations, numbers, or clauses.\n"
            "7. Do not summarize, condense, or simplify the text.\n"
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
            raw = response["message"]["content"].strip()
            return clean_translation_meta(raw)
        except Exception as e:
            print(f"Translation Error ({target_lang}): {e}")
            return text
