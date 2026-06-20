#!/usr/bin/env python3
"""
macOS `say` TTS — zero install, async, stoppable (for barge-in).

speak() spawns `say` as a subprocess and returns an estimated duration so the
orchestrator can time the SPEAKING state; stop() terminates it instantly when
the user barges in. The robot's own speech can also be tee'd into predictor
audio ch1 (DESIGN §1) — see service.py.
"""
from __future__ import annotations
import shutil
import subprocess
from interfaces import TTS


class SayTTS(TTS):
    def __init__(self, voice: str | None = None, rate_wpm: int = 185):
        if shutil.which("say") is None:
            raise RuntimeError("`say` not found (macOS only). Use another TTS backend.")
        self.voice = voice
        self.rate = rate_wpm
        self.proc: subprocess.Popen | None = None

    def speak(self, text: str) -> float:
        self.stop()
        cmd = ["say", "-r", str(self.rate)]
        if self.voice:
            cmd += ["-v", self.voice]
        cmd.append(text)
        self.proc = subprocess.Popen(cmd)
        words = max(1, len(text.split()))
        return words / self.rate * 60.0          # estimated duration (s)

    def stop(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.terminate()
        self.proc = None

    def is_done(self) -> bool:
        return self.proc is None or self.proc.poll() is not None


if __name__ == "__main__":
    import time
    t = SayTTS()
    d = t.speak("Hello. The turn taking gate is now wired to text to speech.")
    print(f"estimated {d:.1f}s; waiting...")
    while not t.is_done():
        time.sleep(0.1)
    print("done.")
