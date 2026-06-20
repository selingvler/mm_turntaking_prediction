#!/usr/bin/env python3
"""
Service-boundary contracts for the LLM-gate orchestrator.

The orchestrator (orchestrator.py) depends only on these abstractions, so the
deterministic stubs are used in tests and the real implementations
(gemini_llm.GeminiLLMClient, tts_say.SayTTS) are swapped in for the live
service — without touching the gate logic.

LLMClient is async/cancellable by contract:
    req = submit(prompt[, t])   # start the call, return a handle
    poll(req[, t]) -> str|None  # None until the response is ready
    cancel(req)                 # abandon a superseded call (refire / barge-in)
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Any, Optional


class LLMClient(ABC):
    @abstractmethod
    def submit(self, prompt: str, t: Optional[float] = None) -> Any: ...
    @abstractmethod
    def poll(self, req: Any, t: Optional[float] = None) -> Optional[str]: ...
    @abstractmethod
    def cancel(self, req: Any) -> None: ...


class TTS(ABC):
    @abstractmethod
    def speak(self, text: str) -> float:
        """Start speaking asynchronously; return an estimated duration (s)."""
    @abstractmethod
    def stop(self) -> None:
        """Stop immediately (barge-in)."""


# ── deterministic stubs (used by orchestrator self-tests) ────────────────

class StubLLM(LLMClient):
    """Ready `latency_s` after submit (by the t passed in), reply = echo."""
    def __init__(self, latency_s: float = 1.2, reply_fn=None):
        self.latency = latency_s
        self.reply = reply_fn or (lambda p: f"<reply to: {p!r}>")
        self._n = 0

    def submit(self, prompt, t=None):
        self._n += 1
        return {"id": self._n, "prompt": prompt, "t0": t, "cancelled": False}

    def poll(self, req, t=None):
        if req["cancelled"]:
            return None
        if req["t0"] is None:
            req["t0"] = t
        if t is None or (req["t0"] is not None and t - req["t0"] >= self.latency):
            return self.reply(req["prompt"])
        return None

    def cancel(self, req):
        req["cancelled"] = True


class StubTTS(TTS):
    def __init__(self, per_char_s: float = 0.04):
        self.per_char = per_char_s
        self.spoken: list = []

    def speak(self, text: str) -> float:
        self.spoken.append(text)
        return max(0.3, len(text) * self.per_char)

    def stop(self) -> None:
        pass
