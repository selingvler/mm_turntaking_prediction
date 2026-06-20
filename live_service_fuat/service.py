#!/usr/bin/env python3
"""
Live LLM-gate voice service — ties the whole pipeline together:

    capture → ShiftPredictor (shift signal) → Orchestrator (gate) → Gemini → TTS

Modes
-----
  --mode console   Type utterances. Real Gemini + macOS `say` TTS + the real
                   orchestrator. No mic/cam needed — validates the gate + Gemini
                   + TTS end-to-end. Each line simulates: user speaks (gate fires
                   the LLM speculatively) then stops (gate commits → speaks).

  --mode live      Mic + webcam drive ShiftPredictor in real time; on a predicted
                   turn-end the gate sends the recent user AUDIO straight to
                   Gemini (multimodal — no separate STT), and speaks the reply
                   once the user actually stops. Always listening (barge-in).

Setup: export GEMINI_API_KEY=...   (live also needs models/face_landmarker.task)
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "live_service"))

from orchestrator import Orchestrator, Config, State          # noqa: E402
from interfaces import LLMClient                              # noqa: E402


# ─────────────────────────────────────────────────────────────────────────
# Console mode: real Gemini + say TTS + real gate, no hardware.
# ─────────────────────────────────────────────────────────────────────────

def run_console():
    from gemini_llm import GeminiLLMClient
    from tts_say import SayTTS
    llm = GeminiLLMClient()
    tts = SayTTS()
    orc = Orchestrator(cfg=Config(thr_now=0.0), llm=llm, tts=tts)  # typed => always "shift"

    print("LLM-gate console. Type a message, Enter to send. Ctrl-C to quit.\n")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print(); return
        if not line:
            continue
        # 1) user is "speaking" and finishing → gate fires Gemini speculatively
        orc.tick(time.time(), score_now=1.0, user_speaking=True, text=line)
        # 2) user stops → poll in real time until commit (SPEAK) or discard
        deadline = time.time() + 20
        spoke = False
        while time.time() < deadline:
            for a in orc.tick(time.time(), 1.0, False, line):
                tag = a["action"]
                if tag == "FIRE_LLM":   print("   · firing Gemini (speculative)…")
                elif tag == "LLM_READY":print("   · Gemini replied; waiting for end-of-turn")
                elif tag == "SPEAK":    print(f"bot> {a['response']}")
                elif tag == "DISCARD_RESPONSE": print("   · response discarded")
            if orc.state == State.SPEAKING:
                spoke = True
                while not tts.is_done():
                    time.sleep(0.05)
                orc.tick(time.time(), 0.0, False, "")   # let TTS_DONE fire
                break
            if orc.state == State.COOLDOWN:
                break
            time.sleep(0.05)
        if not spoke and orc.state != State.SPEAKING:
            # ensure we don't wedge; reset to listening
            orc.state = State.LISTENING; orc.text = ""


# ─────────────────────────────────────────────────────────────────────────
# Live mode: mic + webcam → ShiftPredictor → gate → Gemini(audio) → TTS.
# ─────────────────────────────────────────────────────────────────────────

class _UtteranceLLM(LLMClient):
    """On fire, transcribe the FULL current-utterance audio FRESH (the live
    continuous transcript lags ~1 STT cycle, so it'd send a half sentence), then
    reply — all in one background task so the loop never blocks. on_text pushes
    the final transcription to the display so HEARD shows exactly what was sent."""
    def __init__(self, gemini, get_wav, on_text=None):
        self.g = gemini; self.get_wav = get_wav; self.on_text = on_text
    def _work(self, wav):
        txt = (self.g.transcribe(wav) or "").strip()
        if self.on_text:
            self.on_text(txt)
        if not txt:
            txt = "(kullanıcı kısa bir şey söyledi)"
        return self.g._call(
            f'Kullanıcı şöyle dedi: "{txt}"\nKısa, sesli okunacak bir yanıt ver.')
    def submit(self, prompt, t=None):
        return {"fut": self.g.pool.submit(self._work, self.get_wav()), "cancelled": False}
    def poll(self, req, t=None): return self.g.poll(req, t)
    def cancel(self, req):       self.g.cancel(req)


class RobustVAD:
    """webrtcvad + an energy gate calibrated to YOUR ambient noise floor.

    speech ⟺ chunk RMS > energy_thr  AND  webrtcvad sees speech in ≥1 frame.
    (The energy lower-bound rejects quiet ambient; webrtcvad rejects loud
    non-speech. No 'sustained speech' requirement, so short words still trigger.
    speech_frac > 0 re-enables a fraction requirement if ever needed.)
    """
    def __init__(self, sr, aggr=3, frame_ms=30, speech_frac=0.0, energy_thr=0.01):
        import webrtcvad, numpy as np
        self._np = np
        self.vad = webrtcvad.Vad(aggr)
        self.sr = sr
        self.flen = int(sr * frame_ms / 1000)           # samples per frame
        self.frac = speech_frac
        self.energy_thr = energy_thr
        self.last_rms = 0.0

    def is_speech(self, x) -> bool:
        np = self._np
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))
        self.last_rms = rms
        if rms < self.energy_thr:                       # below noise floor → silence
            return False
        pcm = (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()
        b = self.flen * 2
        tot = sp = 0
        for o in range(0, len(pcm) - b + 1, b):
            tot += 1
            if self.vad.is_speech(pcm[o:o + b], self.sr):
                sp += 1
        if tot == 0:
            return False
        return sp >= 1 if self.frac <= 0 else (sp / tot) >= self.frac


class CameraThread:
    """Capture + MediaPipe 60-dim features in the BACKGROUND so the prediction
    loop never blocks. Each feature is TIME-STAMPED, so window() can resample to a
    real-time 30 fps grid regardless of the achieved camera fps — keeping video
    time-aligned with audio (otherwise a slow camera makes 'now' lag, and p_now
    rises only after you've stopped)."""
    def __init__(self, feats, device=0, maxlen=1200):
        import cv2, threading
        from collections import deque
        self.feats = feats
        self.cam = cv2.VideoCapture(device)
        self.buf = deque(maxlen=maxlen)
        self.ts = deque(maxlen=maxlen)
        self.lock = threading.Lock()
        self._stop = threading.Event()
        self.fps = 0.0
        self.opened = self.cam.isOpened()
        self._th = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._th.start(); return self

    def _run(self):
        import time as _t
        n, t0 = 0, _t.time()
        while not self._stop.is_set():
            ok, frame = self.cam.read()
            if not ok:
                _t.sleep(0.01); continue
            f = self.feats.frame_to_60(frame)
            with self.lock:
                self.buf.append(f); self.ts.append(_t.time())
            n += 1; dt = _t.time() - t0
            if dt >= 1.0:
                self.fps = n / dt; n = 0; t0 = _t.time()

    def window(self, now: float, dur_s: float, fps_out: int):
        """Return (dur_s*fps_out, 60) features on a real-time grid ending at `now`,
        nearest-previous sample per grid point (causal). Zeros if no frames yet."""
        import numpy as np
        k = int(round(dur_s * fps_out))
        with self.lock:
            if not self.buf:
                return np.zeros((k, 60), np.float32)
            ts = np.asarray(self.ts); fe = np.stack(self.buf).astype(np.float32)
        grid = np.linspace(now - dur_s, now, k)
        idx = np.clip(np.searchsorted(ts, grid, side="right") - 1, 0, len(fe) - 1)
        return fe[idx]

    def release(self):
        self._stop.set(); self.cam.release()


def run_live(thr_now: float, hop_s: float, window_s: float, weights: str | None,
             transcribe: bool = False, vad_aggr: int = 3, vad_frac: float = 0.0,
             noise_margin: float = 3.0, energy_thr: float | None = None,
             mic_device=None, log_path: str | None = None, gui_port: int = 8765,
             audio_only: bool = False, signal: str = "p_future"):
    import numpy as np, sounddevice as sd, io, wave, threading, json
    from predictor import ShiftPredictor, AUDIO_SR, VIDEO_FPS
    from live_video import LiveFaceFeatures
    from gemini_llm import GeminiLLMClient
    from tts_say import SayTTS
    from webgui import WebGUI

    SR = AUDIO_SR
    pred = ShiftPredictor(weights_path=weights, window_s=window_s,
                          fill_std=0.01, thr_now=thr_now, audio_only=audio_only,
                          signal=signal)
    print(f"signal = {signal}   fire threshold = {thr_now}")
    feats = None if audio_only else LiveFaceFeatures()  # mediapipe 60-dim/frame
    if audio_only:
        print("--audio-only: using VAP_candor audio model (no camera / MediaPipe)")
    tts = SayTTS()

    # ── calibrate ambient noise floor (stay silent ~1.5 s) ───────────────
    if energy_thr is None:
        print("Calibrating ambient noise (~1.5 s) — please stay SILENT…")
        amb = sd.rec(int(1.5 * SR), samplerate=SR, channels=1, dtype="float32",
                     device=mic_device)
        sd.wait()
        noise_rms = float(np.sqrt(np.mean(amb ** 2) + 1e-12))
        energy_thr = max(0.012, noise_rms * noise_margin)
        print(f"  noise_rms={noise_rms:.4f}  →  energy_thr={energy_thr:.4f} "
              f"(override with --energy-thr)")
    vad = RobustVAD(SR, aggr=vad_aggr, speech_frac=vad_frac, energy_thr=energy_thr)

    # rolling raw mic = the full predictor window (kept real-time fresh)
    RING_S = max(20, int(round(window_s)))
    mic_ring = np.zeros(SR * RING_S, np.float32); mic_lock = threading.Lock()
    speaking_flag = {"v": False}

    # SINGLE current-utterance boundary: audio sent to the LLM is only since the
    # last committed reply, so a previously-answered utterance never leaks in.
    # Reset to "now" on every SPEAK (see the loop).
    utt_start = {"v": time.time()}

    def wav_bytes():
        with mic_lock:
            n = int((time.time() - utt_start["v"]) * SR)
            n = max(SR // 2, min(len(mic_ring), n))   # ≥0.5 s, ≤ ring length
            pcm = (np.clip(mic_ring[-n:], -1, 1) * 32767).astype("<i2").tobytes()
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
        return buf.getvalue()

    gem = GeminiLLMClient()
    transcript = {"v": ""}        # live STT of the CURRENT utterance (continuous)
    orc = Orchestrator(cfg=Config(thr_now=thr_now, min_speak_s=0.8),
                       llm=_UtteranceLLM(gem, get_wav=wav_bytes,
                                         on_text=lambda x: transcript.__setitem__("v", x)),
                       tts=tts)

    # mic callback: fill ring + robust VAD (energy floor + sustained speech)
    def on_audio(indata, frames, tinfo, status):
        x = indata[:, 0].astype(np.float32)
        with mic_lock:
            n = len(x); mic_ring[:-n] = mic_ring[n:]; mic_ring[-n:] = x
        speaking_flag["v"] = vad.is_speech(x)

    cam = None if audio_only else CameraThread(feats, device=0).start()
    if cam is not None and not cam.opened:
        print("⚠️  Webcam (device 0) did not open — video features will be zeros.")

    # CONTINUOUS live STT of the current utterance (always on — it IS the text we
    # send). Only runs while there's speech since the last reset, so silence isn't
    # transcribed. transcript is cleared on SPEAK (new turn from zero).
    stop = threading.Event()
    spoke = {"v": False}            # has the user spoken since the last reset?

    def _tloop():
        while not stop.is_set():
            if spoke["v"]:
                txt = gem.transcribe(wav_bytes())
                if txt:
                    transcript["v"] = txt
            stop.wait(0.6)
    threading.Thread(target=_tloop, daemon=True).start()

    gui = WebGUI(port=gui_port).start()
    logf = open(log_path, "w") if log_path else None

    def jlog(obj):
        if logf:
            logf.write(json.dumps(obj) + "\n"); logf.flush()

    EV_KIND = {"FIRE_LLM": "fire", "CANCEL_LLM": "fire", "LLM_READY": "ready",
               "SPEAK": "speak", "TTS_DONE": "info", "ENTER_GRACE": "info",
               "DISCARD_RESPONSE": "bad", "BARGE_IN_STOP_TTS": "bad"}

    t_start = time.time()
    last_reply = {"v": ""}
    fire_t = {"v": None}
    speak_since = {"v": None}
    last_audio_s = {"v": 0.0}        # duration of the last/ongoing utterance
    prev_spk = {"v": False}
    utt_peak = {"v": 0.0}            # max shift score during current utterance
    utt_end = {"v": 0.0}            # shift score in the last speaking frame
    hop_hz = 0.0

    win_a = int(round(window_s * SR))
    with sd.InputStream(channels=1, samplerate=SR, device=mic_device,
                        blocksize=int(0.05 * SR), callback=on_audio):   # 50 ms = low latency
        try:
            while True:
                loop0 = time.time()
                now = time.time()
                # build the FULL window, time-aligned to NOW (no push drift):
                with mic_lock:
                    mono = mic_ring[-win_a:].copy()       # last window_s of real audio
                aw = np.empty((mono.shape[0], 2), np.float32)
                aw[:, 0] = mono                           # ch0 = human
                aw[:, 1] = np.random.randn(mono.shape[0]).astype(np.float32) * 0.01  # ch1 robot
                if cam is not None:
                    vw = cam.window(now, window_s, VIDEO_FPS)        # (600,60) real-time grid
                    vfull = np.zeros((vw.shape[0], 60, 2), np.float32)
                    vfull[:, :, 0] = vw                   # ch0 = human face; ch1 = zeros
                else:
                    vfull = None                          # audio-only model ignores video

                p0 = time.time()
                out = pred.predict_window(aw, vfull)
                predict_ms = (time.time() - p0) * 1000
                score = float(out["score_now"])

                t = time.time() - t_start
                spk = speaking_flag["v"]
                if spk:
                    speak_since["v"] = speak_since["v"] or t
                    last_audio_s["v"] = t - speak_since["v"]
                    utt_peak["v"] = max(utt_peak["v"], score)
                    utt_end["v"] = score
                    spoke["v"] = True              # there is speech to transcribe
                else:
                    speak_since["v"] = None

                # turn-end telemetry: report the score you reached WHILE speaking
                # vs after — the numbers you need to pick a threshold.
                if prev_spk["v"] and not spk:
                    gui.event(t, "TURN_END",
                              f"peak {signal}[robot]={utt_peak['v']:.3f} · "
                              f"last-while-speaking={utt_end['v']:.3f} · now(silent)={score:.3f}",
                              "info")
                    jlog({"t": round(t, 2), "event": "TURN_END",
                          "peak_score": round(utt_peak["v"], 4),
                          "end_score": round(utt_end["v"], 4),
                          "silent_score": round(score, 4)})
                    utt_peak["v"] = 0.0
                prev_spk["v"] = spk

                text = "[user speech]" if spk else orc.text

                for a in orc.tick(t, score, spk, text):
                    act = a["action"]
                    if act == "FIRE_LLM":
                        fire_t["v"] = t
                        msg = "(transcribing full utterance…)"   # actual text set at LLM_READY
                        trig = a.get("trigger", "?")
                        detail = (f"trigger={trig}  ·  sent → {msg}")
                        gui.event(a["t"], "FIRE_LLM", detail, "fire")
                        jlog({"t": a["t"], "event": "FIRE_LLM", "trigger": trig,
                              "message": msg, "score": round(score, 4)})
                    elif act == "LLM_READY":
                        gui.event(a["t"], "LLM_READY",
                                  f"latency={a.get('latency')}s  ·  asked: \"{transcript['v']}\""
                                  f"  ·  reply: {a.get('response','')}", "ready")
                        jlog({"t": a["t"], "event": "LLM_READY", "latency": a.get("latency"),
                              "asked": transcript["v"], "reply": a.get("response", "")})
                    elif act == "SPEAK":
                        last_reply["v"] = a.get("response", ""); transcript["v"] = ""
                        fire_t["v"] = None
                        utt_start["v"] = time.time()   # clear recording: new turn from zero
                        spoke["v"] = False             # stop transcribing until next speech
                        gui.event(a["t"], "SPEAK", a.get("response", ""), "speak")
                        jlog({"t": a["t"], "event": "SPEAK", "reply": a.get("response", "")})
                    else:
                        if act == "DISCARD_RESPONSE":
                            fire_t["v"] = None
                        if act == "BARGE_IN_STOP_TTS":
                            utt_start["v"] = time.time()   # interruption = new turn
                        gui.event(a["t"], act, a.get("reason", ""), EV_KIND.get(act, "info"))
                        jlog({"t": a["t"], "event": act, "reason": a.get("reason", "")})

                waiting = orc._req is not None and orc._response is None
                llm = ("✓ reply ready" if orc._response is not None
                       else (f"⏳ waiting {t - fire_t['v']:.1f}s"
                             if (waiting and fire_t["v"]) else "idle"))
                gui.set_user(camera_fps=(cam.fps if cam else 0.0), mic_rms=vad.last_rms,
                             mic_gate=vad.energy_thr, speaking=spk,
                             audio_s=last_audio_s["v"] if (spk or waiting) else 0.0,
                             transcript=transcript["v"] or "…")
                gui.set_service(state=orc.state.name, score=score, thr=thr_now,
                                score_user=out["score_user"], signal=signal,
                                llm=llm, last_reply=last_reply["v"],
                                predict_ms=predict_ms, hop_hz=hop_hz, softmax=2.0)
                jlog({"t": round(t, 2), "tick": True, "state": orc.state.name,
                      "speaking": spk, "score_robot": round(score, 4),
                      "score_user": round(float(out["score_user"]), 4),
                      "mic_rms": round(vad.last_rms, 5), "cam_fps": round(cam.fps if cam else 0.0, 1),
                      "predict_ms": round(predict_ms, 1), "hop_hz": round(hop_hz, 2)})

                dt = time.time() - loop0
                time.sleep(max(0, hop_s - dt))
                hop_hz = 1.0 / max(1e-3, time.time() - loop0)
        except KeyboardInterrupt:
            pass
        finally:
            stop.set();
            if cam: cam.release()
            gui.stop()
            if logf: logf.close()
    print("bye.")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["console", "live"], default="console")
    ap.add_argument("--thr",    type=float, default=None,
                    help="fire threshold (default: p_future→0.5, p_now→0.05; tune from TURN_END)")
    ap.add_argument("--hop",    type=float, default=0.5)
    ap.add_argument("--window", type=float, default=20.0)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--transcribe", action="store_true",
                    help="live mode: show a Gemini transcript of what you said (extra API calls)")
    ap.add_argument("--vad-aggr", type=int, default=3, choices=[0, 1, 2, 3],
                    help="webrtcvad aggressiveness (3 = most aggressive at rejecting non-speech)")
    ap.add_argument("--vad-frac", type=float, default=0.0,
                    help="0 = any speech frame triggers (default; good for short words). "
                         ">0 requires that fraction of frames (stricter, may drop short speech)")
    ap.add_argument("--noise-margin", type=float, default=3.0,
                    help="energy_thr = ambient_rms * this (raise if noise still triggers)")
    ap.add_argument("--energy-thr", type=float, default=None,
                    help="override the calibrated energy threshold directly")
    ap.add_argument("--mic-device", type=int, default=None, help="input device index")
    ap.add_argument("--log", default=None,
                    help="write JSONL event+telemetry log here (paste it back for analysis)")
    ap.add_argument("--gui-port", type=int, default=8765, help="web GUI port")
    ap.add_argument("--audio-only", action="store_true",
                    help="use the audio-only VAP_candor model (no camera/MediaPipe)")
    ap.add_argument("--signal", choices=["p_now", "p_future"], default="p_future",
                    help="VAP horizon driving the score (p_future = 0.6–2.0s ahead)")
    a = ap.parse_args()
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("Set GEMINI_API_KEY first:  export GEMINI_API_KEY=...")
    thr = a.thr if a.thr is not None else (1.5 if a.signal == "p_future" else 0.05)
    if a.mode == "console":
        run_console()
    else:
        run_live(thr, a.hop, a.window, a.weights, transcribe=a.transcribe,
                 vad_aggr=a.vad_aggr, vad_frac=a.vad_frac,
                 noise_margin=a.noise_margin, energy_thr=a.energy_thr,
                 mic_device=a.mic_device, log_path=a.log, gui_port=a.gui_port,
                 audio_only=a.audio_only, signal=a.signal)


if __name__ == "__main__":
    main()
