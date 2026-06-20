#!/usr/bin/env python3
"""
Live terminal dashboard for the LLM-gate service (rich).

A single self-updating panel that always shows, at a glance:
  STATE   — the orchestrator state (colour-coded)
  USER    — speaking or silent (mic VAD)
  SHIFT   — the live shift score vs the fire threshold, with a bar + FIRE flag
  LLM     — idle / waiting (with elapsed) / ready
  AUDIO   — seconds of speech buffered for the prompt
  HEARD   — current transcript (if --transcribe), i.e. the "stored STT text"
  BOT     — the last spoken reply
plus a short scrolling event log.

Self-test (renders a few static frames, no terminal needed):
    python live_service/dashboard.py
"""
from __future__ import annotations
import time
from collections import deque

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_STATE_STYLE = {
    "LISTENING":   ("cyan",    "👂 LISTENING"),
    "SPECULATING": ("yellow",  "⚡ SPECULATING (LLM firing)"),
    "GRACE_WAIT":  ("magenta", "⏳ GRACE (reply ready, waiting for end-of-turn)"),
    "SPEAKING":    ("bold green", "🔊 SPEAKING"),
    "COOLDOWN":    ("dim",     "💤 COOLDOWN"),
}
_ACTION_STYLE = {
    "FIRE_LLM": "yellow", "CANCEL_LLM": "yellow", "LLM_READY": "blue",
    "ENTER_GRACE": "magenta", "SPEAK": "bold green", "TTS_DONE": "green",
    "DISCARD_RESPONSE": "red", "BARGE_IN_STOP_TTS": "bold red",
}


def _bar(value: float, thr: float, width: int = 24, soft_max: float = 2.0) -> Text:
    frac = max(0.0, min(1.0, value / soft_max))
    fired = value > thr
    filled = int(frac * width)
    t = Text()
    t.append("█" * filled, style="bold green" if fired else "cyan")
    t.append("░" * (width - filled), style="dim")
    t.append(f"  {value:5.3f}", style="bold" if fired else "")
    t.append(f"  (thr {thr:.3f})", style="dim")
    if fired:
        t.append("  ● FIRE", style="bold green")
    return t


class Dashboard:
    def __init__(self, title="MM-VAP Turn-Taking LLM Gate"):
        self.title = title
        self.events: deque = deque(maxlen=8)
        self.console = Console()
        self.live = Live(self._render({}), console=self.console,
                         refresh_per_second=10, screen=False)

    def __enter__(self):
        self.live.start()
        return self

    def __exit__(self, *a):
        self.live.stop()

    def push_event(self, t: float, action: str, detail: str = ""):
        style = _ACTION_STYLE.get(action, "white")
        line = Text(f"t={t:6.1f}  ", style="dim")
        line.append(action, style=style)
        if detail:
            line.append(f"  {detail}", style="dim")
        self.events.appendleft(line)

    def update(self, **s):
        self.live.update(self._render(s))

    def _render(self, s) -> Group:
        state = s.get("state", "—")
        style, label = _STATE_STYLE.get(state, ("white", state))

        tbl = Table.grid(padding=(0, 2))
        tbl.add_column(justify="right", style="bold dim", min_width=7)
        tbl.add_column()

        tbl.add_row("STATE", Text(label, style=style))

        spk = s.get("user_speaking")
        user = (Text("● SPEAKING", style="bold green") if spk
                else Text("○ silent", style="dim"))
        tbl.add_row("USER", user)

        if "mic_rms" in s:
            rms = s["mic_rms"]; et = s.get("energy_thr", 0.0)
            over = rms >= et
            m = Text()
            m.append(f"{rms:.4f}", style="green" if over else "dim")
            m.append(f"  (gate {et:.4f})", style="dim")
            m.append("  ▲ over" if over else "  ▽ under",
                     style="green" if over else "dim")
            tbl.add_row("MIC", m)

        if "score" in s:
            tbl.add_row("SHIFT", _bar(s.get("score", 0.0), s.get("thr", 0.015)))

        why = s.get("why")
        if why:
            wstyle = ("bold green" if "FIRE" in why else
                      "dim" if "silent" in why else "yellow")
            tbl.add_row("FIRE?", Text(why, style=wstyle))

        if s.get("llm_ready"):
            llm = Text("✓ reply ready", style="blue")
        elif s.get("llm_waiting"):
            el = s.get("llm_elapsed", 0.0)
            llm = Text(f"⏳ waiting… {el:.1f}s", style="yellow")
        else:
            llm = Text("idle", style="dim")
        tbl.add_row("LLM", llm)

        if "audio_s" in s:
            tbl.add_row("AUDIO", Text(f"{s['audio_s']:.1f}s buffered", style="dim"))

        if "cam_fps" in s or "predict_ms" in s or "hop_hz" in s:
            cam = s.get("cam_fps", 0.0); pms = s.get("predict_ms", 0.0)
            hz = s.get("hop_hz", 0.0)
            perf = Text()
            perf.append(f"cam {cam:4.1f} fps", style="green" if cam >= 20 else "yellow")
            perf.append("   ")
            perf.append(f"predict {pms:5.0f} ms", style="green" if pms < 300 else "yellow")
            perf.append("   ")
            perf.append(f"loop {hz:4.1f} Hz", style="green" if hz >= 1.5 else "yellow")
            tbl.add_row("PERF", perf)

        heard = s.get("transcript")
        if heard is not None:
            tbl.add_row("HEARD", Text(heard or "…", style="white"))

        bot = s.get("last_reply")
        if bot:
            tbl.add_row("BOT", Text(bot, style="bold green"))

        panel = Panel(tbl, title=self.title, border_style=style.split()[-1])

        log = Table.grid()
        log.add_column()
        log.add_row(Text("recent events", style="dim"))
        for line in self.events:
            log.add_row(line)
        return Group(panel, log)


def _selftest():
    """Render a scripted sequence so we can see it works without a terminal app."""
    import itertools
    frames = [
        dict(state="LISTENING", user_speaking=True, score=0.004, thr=0.015,
             audio_s=1.2, transcript="", last_reply=""),
        dict(state="LISTENING", user_speaking=True, score=0.31, thr=0.015,
             audio_s=2.4, transcript="merhaba nasıl", last_reply=""),
        dict(state="SPECULATING", user_speaking=True, score=0.42, thr=0.015,
             llm_waiting=True, llm_elapsed=0.3, audio_s=3.1,
             transcript="merhaba nasılsın", last_reply=""),
        dict(state="GRACE_WAIT", user_speaking=False, score=0.05, thr=0.015,
             llm_ready=True, audio_s=3.4, transcript="merhaba nasılsın", last_reply=""),
        dict(state="SPEAKING", user_speaking=False, score=0.0, thr=0.015,
             audio_s=0.0, transcript="merhaba nasılsın", last_reply="İyiyim, teşekkürler."),
    ]
    evs = ["FIRE_LLM", "LLM_READY", "ENTER_GRACE", "SPEAK"]
    with Dashboard() as d:
        for i, fr in enumerate(frames):
            if i < len(evs):
                d.push_event(i * 0.5, evs[i])
            d.update(**fr)
            time.sleep(0.8)
    print("dashboard self-test done.")


if __name__ == "__main__":
    _selftest()
