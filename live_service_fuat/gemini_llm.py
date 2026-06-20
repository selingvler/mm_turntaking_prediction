#!/usr/bin/env python3
"""
Real cancellable Gemini LLM client for the orchestrator (google.genai SDK).

Each submit() launches the call on a thread pool and returns a handle; poll()
returns the text once done, None meanwhile; cancel() abandons a superseded call
(refire / barge-in / discard). The HTTP request can't always be killed, but the
result is dropped so it never reaches the user.

Gemini is multimodal, so `contents` can be a string OR a list mixing text and
audio parts — the live service can hand it the user's audio directly, no separate
STT needed (see service.py).

Env: GEMINI_API_KEY (required), GEMINI_MODEL (default gemini-2.0-flash).
"""
from __future__ import annotations
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

from interfaces import LLMClient

DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")
DEFAULT_SYSTEM = (
    "Canlı bir sesli konuşmada bulunan, arkadaş canlısı ve kısa yanıtlar veren bir sesli asistansın. "
    "Yüksek sesle okunmaya uygun, bir veya iki kısa cümleyle yanıt ver. "
    "Markdown, liste veya emoji kullanma."
)


class GeminiLLMClient(LLMClient):
    def __init__(self, model: str = DEFAULT_MODEL, api_key: Optional[str] = None,
                 system: Optional[str] = DEFAULT_SYSTEM, max_workers: int = 4):
        from google import genai
        from google.genai import types
        self._types = types
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("GEMINI_API_KEY not set (export it or pass api_key=).")
        self.client = genai.Client(api_key=key)
        self.model = model
        self.system = system
        self.pool = ThreadPoolExecutor(max_workers=max_workers)

    def _call(self, contents) -> str:
        # Low-latency config for a voice assistant: disable 2.5 "thinking"
        # (it adds seconds of latency) and cap output length.
        kw = dict(max_output_tokens=120)
        if self.system:
            kw["system_instruction"] = self.system
        try:
            kw["thinking_config"] = self._types.ThinkingConfig(thinking_budget=0)
        except Exception:
            pass
        cfg = self._types.GenerateContentConfig(**kw)
        r = self.client.models.generate_content(
            model=self.model, contents=contents, config=cfg)
        return (r.text or "").strip()

    # contents may be a str or a multimodal list (text + audio Parts)
    def submit(self, prompt: Any, t: Optional[float] = None) -> Any:
        return {"fut": self.pool.submit(self._call, prompt), "cancelled": False}

    def poll(self, req: Any, t: Optional[float] = None) -> Optional[str]:
        if req["cancelled"]:
            return None
        fut = req["fut"]
        if fut.done():
            try:
                return fut.result()
            except Exception as e:                       # surface, don't crash the loop
                return f"[LLM error: {e}]"
        return None

    def cancel(self, req: Any) -> None:
        req["cancelled"] = True
        req["fut"].cancel()

    def audio_part(self, pcm_bytes: bytes, mime: str = "audio/wav"):
        """Wrap raw audio bytes as a Gemini Part for multimodal prompts."""
        return self._types.Part.from_bytes(data=pcm_bytes, mime_type=mime)

    def transcribe(self, wav_bytes: bytes, language: str = "Turkish") -> str:
        """Verbatim transcription for DISPLAY (no assistant persona)."""
        try:
            cfg = self._types.GenerateContentConfig(max_output_tokens=120)
            try:
                cfg.thinking_config = self._types.ThinkingConfig(thinking_budget=0)
            except Exception:
                pass
            r = self.client.models.generate_content(
                model=self.model,
                contents=[f"The speech is in {language}. Transcribe it verbatim in "
                          f"{language}. Output only the spoken words, nothing else. "
                          f"If there is no clear speech, output nothing.",
                          self.audio_part(wav_bytes)],
                config=cfg)
            return (r.text or "").strip()
        except Exception:
            return ""
