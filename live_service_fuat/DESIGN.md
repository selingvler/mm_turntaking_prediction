# Live Turn-Shift Prediction Service — Design

A real-time gate that sits in front of an LLM voice assistant. It uses the VAP
turn-taking model to **speculatively start the LLM before the user finishes
speaking**, then decides whether to play the response based on what the user
actually did. The robot **never stops listening**.

---

## 0. Model facts that constrain the design

From `vap_tasks_explained.md`, `selinc/run_early_fusion.py`, and the checkpoint config:

- `EarlyVAFusion` is a transformer but **the window length is NOT fixed.** `GPT`/`GPTStereo`
  use **ALiBi** attention (no positional embeddings, no max-sequence-length, model.py:17);
  `EncoderCPC` is fully convolutional (fixed 320× downsample, any length); the video
  upsampler uses `scale_factor = 1000/600 = 1.667` as a **ratio**, and `EarlyVAFusion.upsample`
  has no `x[:, :1000]` truncation. So `sequence_len: 1000`/`600` only derive that ratio.
  The 20 s window in `run_early_fusion.py` is a training-matched **choice**, not a requirement —
  we can slide a shorter window (see §2). The one hard rule is alignment: keep
  **audio @ 16 kHz + video @ 30 fps** so upsampled-video length matches audio-embedding length
  (at any window W: audio→W·50 frames, video W·30→×1.667→W·50 ✓).
- Output: 50 fps. Per frame → softmax over 256 states → `p_now` (0–0.6 s) and
  `p_future` (0.6–2.0 s), each `[2]` (per speaker, sums to 1).
- **Shift score** for the human (channel 0): `score = p_*[robot_channel] = p_*[1]`.
  High → model expects the human to yield the floor.
- Validation: `p_now` more accurate (s_pred bal-acc 0.93), `p_future` earlier but
  noisier (0.84). We exploit **both** — see §3.
- Use **fixed thresholds** from `selinc/validate.py::AVG_THRESHOLDS_EP0`, never the
  validation-time optimal ones (those leak; see vap_tasks_explained §7).

### §0a. The model is fully causal (verified)

`GPT`/`GPTStereo` use `MultiHeadAttentionAlibi`, whose `get_alibi_mask` always adds a
lower-triangular causal mask (modules.py:179-186). This applies to **self-attention and
cross-attention** (speaker↔speaker, audio↔video) alike — every attention call passes
`mask=None`, so every one is causal. A frame attends only to frames ≤ itself; there is no
future leakage. **This is what makes real-time operation sound**: the present frame's
output is identical whether computed live (window right-edge) or offline (mid-window), so
offline-calibrated thresholds and behaviour carry over directly. See §2 for consequences.

---

## 1. The single-human / two-channel problem

The model expects stereo (2 speakers). We have one human.

**Decision:**
- **Audio ch0 = human mic.** **Audio ch1 = robot.**
  - While listening: feed **low-amplitude Gaussian noise (std ≈ 0.01)** on ch1 — NOT
    digital zeros (those are OOD for the frozen CPC encoder). The exact value **must
    equal the `SILENCE_FILL_STD` used in fine-tuning** (§8): the model only behaves
    correctly on the silent channel if train and deploy feed the identical fill.
    Stays at ~0.01 through load (peak-norm only triggers >0.05).
  - While the robot speaks (TTS): feed the **actual TTS waveform** into ch1 (barge-in).
- **Video ch0 = human face features (60-dim, as in `build_60feat_pkl`).**
  **Video ch1 = robot:** zeros (matching `SILENCE_VIDEO=1` in fine-tuning). The robot
  has no camera face; train and deploy must agree here too.

**The core bet — TESTED, PASSES (see §8).** Even with ch1 silent the model DOES raise
`p[1]` at the human's end-of-turn. Measured correctly (only handovers *toward* the silent
channel, `validate.py --incoming-channel 1`), pretrained `s_pred_p_now` AUC = **0.94** with
ch1 silenced — as good as the clean baseline (0.88). The earlier "0.49 collapse" was a
direction-contamination artifact: on 2-human test audio the pooled metric also scored 1→0
shifts, where speaker-1's real speech had been silenced (nonsense). **Consequence: the
pretrained Candor weights already drive single-human shift prediction; fine-tuning with
silenced-partner augmentation (§8) is a refinement (helps the reactive shift_hold/gap_0
signals), not a prerequisite.** Caveat: small n (~40) and still a 2-human proxy — a real
single-human-to-robot clip is the gold-standard confirmation.

---

## 2. Streaming inference

**The model is fully causal (§0a), which decides the whole live strategy.** Self- AND
cross-attention apply a `tril` mask (modules.py:179-186), so a frame attends only to
frames ≤ itself. Therefore the right-edge "present" frame of a sliding window sees
exactly the past it would mid-window offline, and never any future →

1. **Take only `output[-1]` each hop** — VALIDATED: the live edge read scores
   `s_pred_p_now` AUC 0.876 (≈ clean baseline 0.88, offline 0.94), and adding lookahead
   setback does NOT help (0.5–2 s all ≤ 0.85). No latency needed. (`eval_streaming.py`)
2. **AUC/ranking transfers, but the EP0 threshold 0.58 does NOT.** The streaming
   single-window read has a lower score *magnitude* than the offline merged timeline
   (fires ~2% at 0.58 vs ~28%). The live operating threshold must be **recalibrated on
   the streaming score distribution** (see `thr*` from eval_streaming.py), not borrowed.
3. **Window length is low-stakes:** ALiBi decays attention to distant past, so the present
   frame is dominated by recent context; 8 s ≈ 20 s for the edge frame.

**Latency (measured, MPS, `bench_latency.py`):** 8 s = 46 ms, 20 s = 132 ms — all far below
a 0.5 s hop. Compute is a non-issue, so default to **`WINDOW_S = 20 s`** (exactly matches
training, zero artifact). Anything ≥ 6–8 s is equally fine.

```
ring buffers (last WINDOW_S s, continuously overwritten); for WINDOW_S = 20:
    audio:  320 000 samples @ 16 kHz, stereo   (ch0 human, ch1 robot/silence)
    video:    600 frames    @ 30 fps, 60-dim, stereo
```

- **Hop:** re-run the model every `HOP_S` (0.5 s; could go to 0.1 s — 10 Hz — and still
  use <50% of budget at 8 s). Keep `output[-1]` as the current present-frame estimate.
- **Warm-up:** until the buffer fills `WINDOW_S`, left-pad with silence / zero frames.
- Run inference in a **dedicated worker thread/process**; capture and STT must never block.
- Fallback (only if ever needed): audio-only `StereoTransformerModel`, already live at
  4 s / 0.5 s in `fuat_scripts/tahmin_fuat.py`.

---

## 3. Trigger: p_now fires, VAD commits (this is the heart of the design)

Two distinct jobs, two signals:

| Stage | Signal | Threshold (EP0) | Action |
|---|---|---|---|
| **FIRE** | rolling **sum of last 10 frames** of `p_now[ROBOT_CHANNEL]` | `s_pred_p_now ≈ 0.58` | Fire the LLM call async with current STT text. |
| **COMMIT** | **external mic VAD** silent ≥ `SILENCE_DEBOUNCE` | — (real audio, not the model) | The turn actually ended → play TTS when the answer is ready. |

**Why p_now, not p_future, to fire:** p_now is far more accurate in validation
(s_pred bal-acc 0.93 vs 0.84 for p_future), so fewer wasted/billable speculative calls
and a cleaner state machine. The cost is less lead time (p_now fires ~0.6 s ahead vs up
to 2 s for p_future). If LLM latency proves too high to hide in 0.6 s, add p_future back
as an *earlier* warm-up trigger — but not in v1.

**Why the 10-frame sum is mandatory, not a per-frame value:** the EP0 thresholds are
calibrated against a 0.2 s (10-frame) summed score (validation.py:578,
`get_spred(p=p_now, window_size=0.2)`), range 0..10. A single frame compared to 0.58 is
meaningless. Live score each hop:

```python
score_now = p_now[-10:, ROBOT_CHANNEL].sum()   # rolling 0.2s, range 0..10
if score_now > THR_NOW:  fire_llm()
```

**Why external VAD for COMMIT, not the model's VAD:** the hard "did the user actually
stop" gate must be low-latency and independent of the 8 s inference cadence — use a
lightweight audio VAD (silero / webrtcvad) on the raw mic. The model's VAD/p_now drives
the *score*; the real mic drives the *commit*.

---

## 4. State machine

States: `LISTENING → SPECULATING → COMMITTED(SPEAKING)`, with grace/discard branches.

```
                    ┌──────────────────────────────────────────────┐
                    │                  LISTENING                    │
                    │  capture A/V + STT; buffer user text;         │
                    │  run predictor every HOP                      │◄────────────┐
                    └──────────────────────────────────────────────┘             │
                             │ p_now sum(10)[1] > THR_NOW                         │
                             ▼                                                    │
                    ┌──────────────────────────────────────────────┐             │
                    │              SPECULATING                      │             │
                    │  fire LLM(text) async; KEEP LISTENING;        │             │
                    │  STT keeps finalizing                         │             │
                    └──────────────────────────────────────────────┘             │
              ┌──────────────┼───────────────────────────┐                       │
              │ user kept    │ mic VAD silent ≥ DEBOUNCE  │ text changed a lot    │
              │ talking >2 s │  (turn really ended)       │ (new shift)           │
              │ no commit    ▼                            ▼                       │
              │     ┌─────────────────────┐    cancel in-flight LLM, refire ──────┤
              │     │   RESPONSE PENDING?  │                                       │
              │     └─────────────────────┘                                       │
              │        │ yes          │ no (LLM still running)                     │
              │        ▼              ▼                                            │
              │   ┌─────────┐   ┌──────────────┐                                   │
              │   │ user    │   │ AWAITING_LLM │── arrives ─► re-check user state  │
              │   │ stopped?│   └──────────────┘                                   │
              │   └─────────┘                                                      │
              │     │yes  │no                                                      │
              │     ▼     ▼                                                        │
              │  SPEAK   GRACE_WAIT(0.5s) ── stopped? ─yes─► SPEAK                 │
              │     │                    └────────── no ──► DISCARD ──────────────►┘ (cooldown)
              │     ▼
              │  ┌──────────────────────────────────┐
              └─►│   SPEAKING (TTS on audio ch1)     │
                 │   still listening for barge-in    │
                 └──────────────────────────────────┘
                     │ TTS done                │ barge-in: human VAD active
                     ▼                         ▼
              reset text buffer,          stop TTS immediately,
              back to LISTENING           treat as new turn → LISTENING
```

### State enum

```python
class State(Enum):
    LISTENING      = auto()   # default; predictor running, text accumulating
    SPECULATING    = auto()   # p_future fired; LLM request in flight; still listening
    AWAITING_LLM   = auto()   # commit conditions met but response not back yet
    GRACE_WAIT     = auto()   # response back but user still talking; ≤0.5 s window
    SPEAKING       = auto()   # TTS playing on ch1; listening for barge-in
    COOLDOWN       = auto()   # after a discard/false-positive; suppress re-trigger briefly
```

---

## 5. Race conditions & how each is handled

1. **Multiple `p_future` triggers before the LLM returns.**
   Keep exactly one in-flight request. On a new trigger, if the finalized transcript
   grew materially (e.g. > N new tokens), **cancel and refire** with the longer text;
   otherwise ignore. Prevents stale, half-sentence prompts.

2. **LLM returns while user is still speaking.**
   → `GRACE_WAIT` 0.5 s. If user stops within it → SPEAK; else **DISCARD** and enter
   `COOLDOWN` so the next `p_future` doesn't instantly re-fire on the same breath.

3. **LLM returns but user already started a *new* utterance** (LLM was slow).
   Response is stale → discard. Detect via "human VAD went silent then active again,
   or transcript diverged from the prompt we sent" → discard.

4. **Barge-in during TTS.** Human VAD active while `SPEAKING` → stop TTS immediately,
   flush the (now-interrupted) response, capture user from zero → LISTENING.

5. **STT lag vs prediction.** The two-stage trigger (§3) is the mitigation: `p_future`
   fires early, but we only SPEAK after `p_now` + silence, by which point STT has
   finalized. Always re-read the finalized transcript at COMMIT; if it differs from the
   speculative prompt, refire (cheap) rather than answer a truncated question.

6. **False-positive shift (user never actually stops).** No COMMIT ever happens; the
   speculative response is discarded; `COOLDOWN` prevents spam; text keeps accumulating
   so a later genuine shift fires on the full utterance.

7. **Speculative LLM cost.** Every `p_future` is a billable call. Use a cancellable /
   streaming client, dedupe by transcript hash, and gate refire on a token-delta
   threshold.

---

## 6. Tunables (single config block)

```python
WINDOW_S         = 20.0     # = training window; free at 132ms (§2). >=6s all fine (causal+ALiBi)
HOP_S            = 0.50     # predictor cadence; measured headroom allows down to ~0.1 s
SCORE_FRAMES     = 10       # p_now rolling-sum window = 0.2s; MUST match EP0 calibration
THR_NOW          = 0.58     # AVG_THRESHOLDS_EP0 ekstedt s_pred_p_now (sum over 10 frames)
SILENCE_DEBOUNCE = 0.30     # s of mic-VAD silence required to call EOT (external VAD)
GRACE_S          = 0.50     # wait window when response arrives mid-speech
COOLDOWN_S       = 1.00     # suppress re-trigger after a discard
REFIRE_TOKEN_DELTA = 4      # new finalized tokens needed to cancel+refire LLM
ROBOT_CHANNEL    = 1        # we read p[ROBOT_CHANNEL] as the shift score
```

All thresholds come from `selinc/validate.py::AVG_THRESHOLDS_EP0` — fixed, not
per-session optimized.

---

## 7. Component interfaces (sketch for `live_service/`)

```python
# interfaces.py
class AudioVideoCapture:   # mic + cam → ring buffers (stereo audio, 60-dim video ch0)
    def latest_window(self) -> tuple[np.ndarray, np.ndarray]: ...   # (320000,2), (600,60,2)
    def write_tts(self, pcm: np.ndarray) -> None: ...               # robot audio → ch1

class STT:                 # streaming; exposes finalized + interim transcript
    def text(self) -> str: ...
    def reset(self) -> None: ...

class ShiftPredictor:      # wraps EarlyVAFusion; device patches from run_early_fusion
    def step(self, audio_win, video_win) -> tuple[float, float, bool]:
        # returns (p_now[1], p_future[1], human_vad_active)

class LLMGate:             # cancellable speculative calls
    def fire(self, prompt: str) -> Request: ...
    def cancel(self, req: Request) -> None: ...

class TTS:
    def speak(self, text: str) -> Stream: ...   # yields PCM → capture.write_tts
    def stop(self) -> None: ...

class Orchestrator:        # owns the State enum + transition table above
    ...
```

---

## 8. Build order (de-risk first)

- [x] **Gate #2 — latency.** PASSED: 8 s = 46 ms, 20 s = 132 ms on MPS (`bench_latency.py`).
- [x] **Windowing/threshold validity.** RESOLVED by causality (§0a) — no experiment needed.
      EP0 thresholds transfer; take `output[-1]`; window length low-stakes.
- [x] **Gate #1 — the channel-1-silent bet (§1). PASSES.** Measured with
      `validate.py --incoming-channel 1` (only handovers toward the silent channel):
      pretrained `s_pred_p_now` AUC = **0.94** with ch1 silenced (clean baseline 0.88).
      The earlier "0.49" was a direction-contamination artifact of the pooled metric on
      2-human audio (it also scored 1→0 shifts where speaker-1's real speech was silenced).
      Always evaluate the silenced case with `--incoming-channel`.

- [~] **Fine-tune with silenced-partner augmentation — optional refinement, not a blocker.**
   `SILENCE_AUG_PROB=0.5 … selinc/finetune_early_fusion.py` (silences a channel at train
   time, keeps labels, masks it out of VAD loss). At epoch 6 it helped the reactive signals
   (gap_0 s_pred 0.91→0.96, shift_hold 0.79→0.86) but the core early `s_pred_p_now` was
   already strong pretrained. Use it to sharpen, not to unblock.

- [x] **Streaming predictor built + validated** (`live_service/predictor.py`,
   `eval_streaming.py`). Live edge read `s_pred_p_now` AUC **0.876** (no lookahead).
   Operating threshold recalibrated to **thr* ≈ 0.015** (F1 0.92) — EP0 0.58 does NOT
   transfer (streaming scale ~40× smaller). Recalibrate per deployment.
- [x] **Orchestrator / LLM-gate state machine built + tested** (`live_service/orchestrator.py`).
   6 states, tick-driven, stubbed LLM/TTS. 5/5 race scenarios pass: happy path,
   false-fire→discard+cooldown, refire-on-longer-transcript, commit-within-grace, barge-in.

- [x] **Service wired (Gemini + TTS).** `interfaces.py` (LLMClient/TTS contracts + stubs),
   `gemini_llm.py` (real cancellable Gemini via google.genai, multimodal — sends user
   AUDIO directly, no separate STT), `tts_say.py` (macOS `say`, stoppable for barge-in),
   `live_video.py` (live 60-dim features reusing build_60feat), `service.py`:
   - `--mode console`: real Gemini + gate + TTS, no hardware — VERIFIED to compile/import;
     run with GEMINI_API_KEY to talk to it by typing.
   - `--mode live`: mic (sounddevice) + webcam (cv2) + webrtcvad → ShiftPredictor → gate →
     Gemini(audio) → say. Built; needs on-device run.

**Remaining (on-device, user-run):**
1. `export GEMINI_API_KEY=…` then `python live_service/service.py --mode console` to
   validate the Gemini gate end-to-end; then `--mode live` for mic/cam.
2. **Live video parity:** `live_video.py` skips build_60feat's per-session normalization
   (no full session live). If live video quality is poor, add online normalization.
3. **Recalibrate `--thr`** on a real session (eval_streaming.py); confirm on a real
   **single-human-to-robot clip** (validation used a 2-human proxy, n~40).
```
