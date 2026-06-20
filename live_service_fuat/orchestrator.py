#!/usr/bin/env python3
"""
LLM-gate orchestrator — the state machine that turns the shift signal into a
voice chatbot that responds fast and ALWAYS listens.

Design (DESIGN.md §3-5). One signal fires speculation, real mic VAD commits:
    FIRE   : score_now (p_now[robot] sum-10, from ShiftPredictor) > THR_NOW
             → start the LLM call early, async, with the current transcript.
    COMMIT : external mic VAD silent ≥ SILENCE_DEBOUNCE (the turn really ended)
             → speak the response once it's back.

It is driven by ticks so the gate logic is testable without real audio/LLM/TTS:
    tick(t, score_now, user_speaking, text) -> list[action]
The LLM and TTS are injected stubs (latency / duration), swap in real ones later.

Races handled (DESIGN §5): refire on longer transcript, response-arrives-mid-
speech → grace → discard, false-fire cooldown, barge-in during TTS, stale
response after a new utterance, multiple in-flight (kept single).

Run the self-checking scenarios:
    python live_service/orchestrator.py
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Optional

from interfaces import LLMClient, TTS, StubLLM, StubTTS


class State(Enum):
    LISTENING   = auto()   # capturing; predictor running; text accumulating
    SPECULATING = auto()   # LLM in flight (fired early); still listening
    GRACE_WAIT  = auto()   # response back but user still talking; ≤ GRACE_S
    SPEAKING    = auto()   # TTS playing; listening for barge-in
    COOLDOWN    = auto()   # after a discard / false fire; suppress re-trigger


@dataclass
class Config:
    thr_now:          float = 0.015   # recalibrated streaming threshold (eval_streaming.py)
    silence_debounce: float = 0.30    # s of mic silence to call end-of-turn
    grace_s:          float = 0.50    # wait window when response arrives mid-speech
    cooldown_s:       float = 1.00    # suppress re-trigger after a discard
    refire_token_delta: int = 3       # new finalized tokens needed to cancel+refire
    min_speak_s:      float = 0.0     # speculative fire needs ≥ this much speech first
                                      # (kills onset fires on a stale silence score)
    llm_latency_s:    float = 1.20    # stub: time from fire to response
    tts_per_char_s:   float = 0.04    # stub: TTS duration model


@dataclass
class Orchestrator:
    cfg: Config = field(default_factory=Config)
    llm: LLMClient = field(default_factory=StubLLM)
    tts: TTS = field(default_factory=StubTTS)
    state:   State = State.LISTENING
    # internal
    text:           str = ""
    _silent_since:  Optional[float] = None
    _req:           Any = None              # in-flight LLM handle
    _req_prompt:    str = ""
    _response:      Optional[str] = None    # ready response awaiting commit
    _cooldown_until: float = -1.0
    _grace_until:   float = -1.0
    _tts_until:     float = -1.0
    _fire_t:        float = -1.0
    _speak_start:   float = -1.0
    _last_prompt_tokens: int = 0
    log: list = field(default_factory=list)

    # ── helpers ──────────────────────────────────────────────────────
    def _emit(self, t, action, **info):
        rec = {"t": round(t, 2), "state": self.state.name, "action": action, **info}
        self.log.append(rec)
        return rec

    def _fire_llm(self, t, reason, trigger="speculative"):
        self._req = self.llm.submit(self.text, t)
        self._req_prompt = self.text
        self._fire_t = t
        self._response = None
        self._last_prompt_tokens = len(self.text.split())
        self.state = State.SPECULATING
        self._emit(t, "FIRE_LLM", reason=reason, trigger=trigger, prompt=self.text)

    def _speak(self, t):
        resp = self._response
        dur = self.tts.speak(resp)
        self._tts_until = t + dur
        self.state = State.SPEAKING
        self._emit(t, "SPEAK", response=resp, until=round(self._tts_until, 2))
        self._req = None
        self._response = None
        self.text = ""                      # reset transcript: new turn from zero
        self._last_prompt_tokens = 0

    def _discard(self, t, reason):
        self._emit(t, "DISCARD_RESPONSE", reason=reason)
        if self._req is not None:
            self.llm.cancel(self._req)
        self._req = None
        self._response = None
        self._cooldown_until = t + self.cfg.cooldown_s
        self.state = State.COOLDOWN

    def _user_stopped(self, t) -> bool:
        return (self._silent_since is not None and
                t - self._silent_since >= self.cfg.silence_debounce)

    # ── main entry: one tick ─────────────────────────────────────────
    def tick(self, t: float, score_now: float, user_speaking: bool, text: str):
        n0 = len(self.log)
        self.text = text                       # we ALWAYS listen: text always updates
        # track silence (commit gate) and speaking-duration (speculative gate)
        if user_speaking:
            self._silent_since = None
            if self._speak_start < 0:
                self._speak_start = t
        else:
            if self._silent_since is None:
                self._silent_since = t
            self._speak_start = -1.0

        # resolve any in-flight LLM that has come back
        if self._req is not None and self._response is None:
            resp = self.llm.poll(self._req, t)
            if resp is not None:
                self._response = resp
                self._emit(t, "LLM_READY", prompt=self._req_prompt,
                           latency=round(t - self._fire_t, 2), response=resp)

        # 0) BARGE-IN always wins: user talks while we speak → stop, listen
        if self.state == State.SPEAKING and user_speaking:
            self.tts.stop()
            self._emit(t, "BARGE_IN_STOP_TTS")
            self._tts_until = -1.0
            self.text = text                   # capture from the interruption on
            self.state = State.LISTENING

        # state machine
        if self.state == State.SPEAKING:
            if t >= self._tts_until:
                self._emit(t, "TTS_DONE")
                self.text = ""
                self.state = State.LISTENING

        elif self.state == State.COOLDOWN:
            if t >= self._cooldown_until:
                self.state = State.LISTENING

        elif self.state == State.LISTENING:
            can = (t >= self._cooldown_until and self.text.strip()
                   and self._req is None)
            spoke_enough = (self._speak_start >= 0
                            and t - self._speak_start >= self.cfg.min_speak_s)
            if can and user_speaking and spoke_enough and score_now > self.cfg.thr_now:
                # (1) PREDICTIVE: p_now crossed thr after sustained speech → fire early
                self._fire_llm(t, reason=f"p_now {score_now:.3f}>thr",
                               trigger="speculative")
            elif can and (not user_speaking) and self._user_stopped(t):
                # (2) FALLBACK: turn ended without a prediction → fire reactively
                self._fire_llm(t, reason="end-of-turn, no prediction",
                               trigger="silence_fallback")

        elif self.state == State.SPECULATING:
            # refire if the finalized transcript grew materially (STT lag / user
            # kept talking) — answer the full question, not a truncated one.
            grew = len(self.text.split()) - self._last_prompt_tokens
            if grew >= self.cfg.refire_token_delta and score_now > self.cfg.thr_now:
                if self._req is not None:
                    self.llm.cancel(self._req)
                self._emit(t, "CANCEL_LLM", reason=f"+{grew} tokens, refire")
                self._fire_llm(t, reason="refire on longer transcript")
            elif self._response is not None:
                if self._user_stopped(t):
                    self._speak(t)             # turn ended + answer ready → speak
                else:
                    self._grace_until = t + self.cfg.grace_s
                    self.state = State.GRACE_WAIT
                    self._emit(t, "ENTER_GRACE", until=round(self._grace_until, 2))

        elif self.state == State.GRACE_WAIT:
            if self._user_stopped(t):
                self._speak(t)
            elif t >= self._grace_until:
                self._discard(t, reason="user still talking after grace")

        return self.log[n0:]


# ── self-checking scenarios (deterministic gate-logic tests) ─────────────

def _run(ticks, cfg=None):
    orc = Orchestrator(cfg=cfg or Config())
    for (t, score, speaking, text) in ticks:
        for a in orc.tick(t, score, speaking, text):
            print(f"    t={a['t']:<5} [{a['state']:<11}] {a['action']}"
                  + (f"  {a.get('reason','')}" if a.get('reason') else ""))
    return [a["action"] for a in orc.log]


def _subseq(actions, expected):
    """expected appears as an ordered subsequence of actions."""
    it = iter(actions)
    return all(any(e == a for a in it) for e in expected)


def main():
    S = 0.5  # tick step
    scenarios = []

    # 1) happy path: fire while talking → user stops → speak
    scenarios.append(("happy path",
        [(0.0,0.0,True,"hello"), (0.5,0.0,True,"hello there"),
         (1.0,0.05,True,"hello there how"), (1.5,0.02,True,"hello there how are you"),
         (2.0,0.0,False,"hello there how are you"), (2.5,0.0,False,"hello there how are you")],
        ["FIRE_LLM","LLM_READY","SPEAK"]))

    # 2) false fire: user never stops → grace expires → discard + cooldown
    scenarios.append(("false fire → discard",
        [(1.0,0.05,True,"I think that"), (1.5,0.04,True,"I think that"),
         (2.0,0.03,True,"I think that"), (2.5,0.02,True,"I think that"),
         (3.0,0.02,True,"I think that"), (3.5,0.02,True,"I think that")],
        ["FIRE_LLM","LLM_READY","ENTER_GRACE","DISCARD_RESPONSE"]))

    # 3) refire when the transcript grows past the token delta
    scenarios.append(("refire on longer transcript",
        [(1.0,0.05,True,"what is"),
         (1.5,0.05,True,"what is the capital of France")],
        ["FIRE_LLM","CANCEL_LLM","FIRE_LLM"]))

    # 4) response arrives mid-speech, user stops within grace → speak
    scenarios.append(("commit within grace",
        [(1.0,0.05,True,"tell me a joke"), (2.0,0.02,True,"tell me a joke"),
         (2.5,0.0,True,"tell me a joke"), (2.6,0.0,False,"tell me a joke"),
         (2.95,0.0,False,"tell me a joke")],
        ["FIRE_LLM","LLM_READY","ENTER_GRACE","SPEAK"]))

    # 5) barge-in: speaking, user talks again → stop TTS → listen
    scenarios.append(("barge-in stops TTS",
        [(1.0,0.05,True,"hi"), (2.5,0.0,False,"hi"), (3.0,0.0,False,"hi"),
         (3.2,0.0,True,"wait no")],   # interrupt during/after speak
        ["SPEAK","BARGE_IN_STOP_TTS"]))

    print("=== orchestrator gate-logic scenarios ===")
    n_pass = 0
    for name, ticks, expect in scenarios:
        print(f"\n[{name}]  expect subsequence: {expect}")
        actions = _run(ticks)
        ok = _subseq(actions, expect)
        print(f"  -> {'PASS' if ok else 'FAIL'}   (saw: {actions})")
        n_pass += ok
    print(f"\n{n_pass}/{len(scenarios)} scenarios passed")
    return 0 if n_pass == len(scenarios) else 1


if __name__ == "__main__":
    raise SystemExit(main())
