#!/usr/bin/env python3
"""
Streaming shift predictor — the core of the live LLM-gate service.

This is the piece every other component (trigger, state machine, LLM gate)
sits on. It wraps EarlyVAFusion and exposes, at each hop, the live FIRE signal:

    score_now = sum over the last SCORE_FRAMES (=0.2s) of p_now[ROBOT_CHANNEL]

calibrated identically to selinc/validate.py (EP0 threshold THR_NOW ≈ 0.58 on a
10-frame sum). Because the model is causal (DESIGN §0a), the most recent frames
of one forward pass over the rolling window are exactly the live present — no
overlap-merge, no recalibration.

Single-human / robot channel (DESIGN §1, validated in gate #1):
    ch0 = human (mic + face features), ch1 = robot.
    While listening, ch1 audio = low Gaussian noise (std FILL_STD, NOT zeros —
    OOD for the frozen CPC encoder), ch1 video = zeros. The shift score reads
    p_now[ch1]: "is the human about to hand the floor to the robot?"

Two entry points
----------------
1. Library: ShiftPredictor — push() human audio/video into the ring buffer,
   call step() each hop. This is what the live service uses.
2. CLI simulation: stream a recording through the predictor as if live, log
   score_now over time, mark ground-truth human turn-ends (0->1 shifts), and
   report catch-rate + a PNG. Doubles as the single-human-clip tester.

    python live_service/predictor.py                       # aeeee708 defaults
    python live_service/predictor.py --weights selinc/runs/<...>/best.pt
    python live_service/predictor.py --audio my.wav --video my.pkl --textgrid my.TextGrid
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "selinc"))

# run_early_fusion installs the CUDA->CPU/MPS patches at import and gives us
# load_model() / load_video() / load_audio_aligned() / _TARGET_DEVICE.
import run_early_fusion as ref  # noqa: E402
from turn_taking.analysis.validation.probabilities import VAPDecoder  # noqa: E402

AUDIO_SR  = 16_000
VIDEO_FPS = 30
LABEL_FPS = 50

VAP_CANDOR_DIR = ROOT / "acl_sample_data_models" / "sample_trained_models" / "VAP_candor"


def _load_audio_model(device):
    """Load the audio-only StereoTransformerModel (VAP_candor) — no video."""
    import yaml
    from turn_taking.model.model import StereoTransformerModel
    cfg_path = VAP_CANDOR_DIR / "20240822_105427_params.yaml"
    w_path   = VAP_CANDOR_DIR / "20240822_105427_fold_0_epoch_10"
    cfg = yaml.safe_load(open(cfg_path))
    ref.fix_config_device(cfg, device)
    m = StereoTransformerModel(cfg=cfg)
    state = torch.load(str(w_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("model."):] if k.startswith("model.") else k): v
             for k, v in state.items()}
    miss, unexp = m.load_state_dict(state, strict=False)
    print(f"[audio-only VAP_candor] missing={len(miss)} unexpected={len(unexp)}")
    return m.to(device).eval()


class ShiftPredictor:
    """Causal streaming predictor over a rolling WINDOW_S window."""

    def __init__(self, weights_path: str | None = None, window_s: float = 20.0,
                 robot_channel: int = 1, fill_std: float = 0.01,
                 score_frames: int = 10, thr_now: float = 0.58,
                 audio_only: bool = False, signal: str = "p_now"):
        self.device   = ref._TARGET_DEVICE
        self.audio_only = audio_only
        assert signal in ("p_now", "p_future")
        self.signal = signal             # which VAP horizon drives the score
        if audio_only:
            self.model = _load_audio_model(self.device)       # StereoTransformerModel
        else:
            if weights_path:
                ref.WEIGHTS_PATH = Path(weights_path)
            self.model = ref.load_model()                     # EarlyVAFusion (A+V)
        self.decoder  = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])

        self.window_s      = window_s
        self.robot_channel = robot_channel
        self.fill_std      = fill_std
        self.score_frames  = score_frames
        self.thr_now       = thr_now

        self.win_audio = int(round(window_s * AUDIO_SR))
        self.win_video = int(round(window_s * VIDEO_FPS))
        # ring buffers, ch0 = human, ch1 = robot (filled on each push)
        self.audio = np.zeros((self.win_audio, 2), dtype=np.float32)
        self.video = np.zeros((self.win_video, 60, 2), dtype=np.float32)

    # ── robot channel synthesis ──────────────────────────────────────
    def _fill_robot_audio(self, n: int) -> np.ndarray:
        if self.fill_std > 0:
            return (np.random.randn(n).astype(np.float32) * self.fill_std)
        return np.zeros(n, dtype=np.float32)

    # ── live ingest ──────────────────────────────────────────────────
    def push(self, human_audio: np.ndarray, human_video: np.ndarray):
        """Append human audio (N,) and video (M,60) to the rolling window;
        the robot channel is synthesised (silent-listener)."""
        n = human_audio.shape[0]
        m = human_video.shape[0]
        if n:
            self.audio = np.roll(self.audio, -n, axis=0)
            self.audio[-n:, 0]                  = human_audio
            self.audio[-n:, self.robot_channel] = self._fill_robot_audio(n)
        if m:
            self.video = np.roll(self.video, -m, axis=0)
            self.video[-m:, :, 0]                  = human_video
            self.video[-m:, :, self.robot_channel] = 0.0

    def step(self) -> dict:
        return self.predict_window(self.audio, self.video)

    # ── core forward ─────────────────────────────────────────────────
    @torch.no_grad()
    def p_now_full(self, audio_win: np.ndarray, video_win: np.ndarray) -> np.ndarray:
        """Forward one window → full per-frame score signal (T,2) numpy at 50 Hz.
        Signal = p_now (0–0.6 s) or p_future (0.6–2.0 s) per self.signal."""
        # Peak-normalize each audio channel to match training (AudioManager /
        # load_audio_aligned: divide by peak when peak > 0.05). Live mic audio is
        # otherwise much quieter than training → OOD for the frozen CPC encoder.
        audio_win = np.ascontiguousarray(audio_win, dtype=np.float32).copy()
        for ch in range(audio_win.shape[1]):
            peak = float(np.abs(audio_win[:, ch]).max())
            if peak > 0.05:
                audio_win[:, ch] /= peak
        a = torch.from_numpy(audio_win).float().unsqueeze(0)
        if self.audio_only:
            _vad, vap = self.model({"audio_chunk": a})
        else:
            f = torch.from_numpy(np.ascontiguousarray(video_win)).float().unsqueeze(0)
            _vad, vap = self.model({"audio_chunk": a, "frames": f})
        vap = torch.softmax(vap, dim=-1).squeeze(0).cpu()          # (T,256)
        p = (self.decoder.p_future(vap) if self.signal == "p_future"
             else self.decoder.p_now(vap))                         # (T,2)
        if p.ndim == 1:
            p = p.unsqueeze(0)
        return p.numpy()

    def score_at(self, p_now: np.ndarray, end_idx: int, ch: int | None = None) -> float:
        """Sum of SCORE_FRAMES of p_now[ch] ending at end_idx (inclusive).
        ch defaults to the robot channel (the shift score)."""
        ch = self.robot_channel if ch is None else ch
        k = self.score_frames
        s = max(0, end_idx - k + 1)
        return float(p_now[s:end_idx + 1, ch].sum())

    def predict_window(self, audio_win: np.ndarray, video_win=None) -> dict:
        """Live read: score at the window right edge (setback=0)."""
        p_now = self.p_now_full(audio_win, video_win)
        last = p_now.shape[0] - 1
        user_ch = 1 - self.robot_channel
        score_now = self.score_at(p_now, last)                       # robot/incoming
        return {
            "score_now":   score_now,                               # p_now[robot] sum-10
            "score_user":  self.score_at(p_now, last, ch=user_ch),  # p_now[you] sum-10
            "fired":       score_now > self.thr_now,
            "p_now_robot": float(p_now[-1, self.robot_channel]),
            "p_now_user":  float(p_now[-1, user_ch]),
        }


# ── file-driven "as-if-live" simulation / single-human-clip tester ───────

def _load_gt_human_turn_ends(textgrid: Path, offset_s: float = 0.0) -> np.ndarray:
    """Times (s) where speaker-0 yields the floor to speaker-1 (0->1 shifts) —
    i.e. the human hands off to the robot. These are when score_now SHOULD spike."""
    from dataset_management.dataset_manager.scripts.get_tt_events import get_events_from_tg
    ev = get_events_from_tg(str(textgrid))
    shifts = ev.get("ekstedt_events", {}).get("shifts", {})
    # get_events_from_tg keys speakers by int (0/1); JSON round-trips to str.
    raw = shifts.get(0, shifts.get("0", []))
    times = np.array([float(t) - offset_s for t in raw], dtype=np.float64)
    return times[times >= 0]


def simulate(audio_path, video_path, textgrid, weights, hop_s, window_s,
             thr_now, fill_std, out_png, max_seconds=None):
    pred = ShiftPredictor(weights_path=weights, window_s=window_s,
                          fill_std=fill_std, thr_now=thr_now)

    print(f"loading video: {video_path}")
    video_arr, vfps = ref.load_video(Path(video_path))           # (T,60,2) @30fps
    dur_s = video_arr.shape[0] / vfps
    print(f"loading audio: {audio_path}")
    audio = ref.load_audio_aligned(Path(audio_path), dur_s, sr_target=AUDIO_SR)

    if max_seconds:
        keep_v = int(round(min(max_seconds, dur_s) * vfps))
        keep_a = int(round(min(max_seconds, dur_s) * AUDIO_SR))
        video_arr = video_arr[:keep_v]; audio = audio[:keep_a]
        dur_s = video_arr.shape[0] / vfps
        print(f"  capped to first {dur_s:.0f}s")

    win_a = int(round(window_s * AUDIO_SR))
    win_v = int(round(window_s * vfps))
    hop_a = int(round(hop_s * AUDIO_SR))
    hop_v = int(round(hop_s * vfps))

    # GT human turn-ends (0->1) to score against
    gt = np.array([])
    if textgrid and Path(textgrid).exists():
        try:
            gt = _load_gt_human_turn_ends(Path(textgrid))
            print(f"GT human turn-ends (0->1 shifts): {len(gt)}")
        except Exception as e:
            print(f"  (could not load GT events: {e})")

    # Sweep how far back from the window edge we read the frame. setback=0 is the
    # true live read (right edge); larger setbacks give the frame more in-window
    # FUTURE context (= added latency), testing the edge-degradation hypothesis
    # (non-causal video upsample + unsupervised last 2 s of training windows).
    SETBACKS_S = [0.0, 0.5, 1.0, 2.0]
    sb_frames  = {sb: int(round(sb * LABEL_FPS)) for sb in SETBACKS_S}
    series = {sb: {"t": [], "s": []} for sb in SETBACKS_S}

    n_steps = max(0, (video_arr.shape[0] - win_v) // hop_v + 1)
    print(f"streaming {n_steps} hops of {hop_s}s over {dur_s:.0f}s "
          f"(window {window_s}s, ch1=noise std {fill_std})")
    print(f"setback sweep (s): {SETBACKS_S}\n")

    for i in range(n_steps):
        va = i * hop_a
        vv = i * hop_v
        a_win = audio[va:va + win_a].copy()
        v_win = video_arr[vv:vv + win_v].copy()
        if a_win.shape[0] < win_a or v_win.shape[0] < win_v:
            break
        a_win[:, pred.robot_channel] = pred._fill_robot_audio(a_win.shape[0])
        v_win[:, :, pred.robot_channel] = 0.0

        p_now = pred.p_now_full(a_win, v_win)                 # (T,2) @50Hz
        T = p_now.shape[0]
        win_start_t = vv / vfps
        for sb, nf in sb_frames.items():
            end_idx = T - 1 - nf
            if end_idx < pred.score_frames:
                continue
            series[sb]["s"].append(pred.score_at(p_now, end_idx))
            series[sb]["t"].append(win_start_t + end_idx / LABEL_FPS)
        if (i + 1) % 50 == 0 or i + 1 == n_steps:
            print(f"  hop {i+1}/{n_steps}  t={win_start_t+window_s:6.1f}s", flush=True)

    # ── catch-rate + separation AUC per setback ──────────────────────
    def catch_and_auc(t_arr, s_arr):
        t_arr = np.array(t_arr); s_arr = np.array(s_arr)
        win = 0.6
        in_rng = [t for t in gt if t_arr.min() <= t <= t_arr.max()]
        caught = sum(np.any((t_arr >= t - win) & (t_arr <= t + win) &
                            (s_arr > thr_now)) for t in in_rng)
        # separation AUC: peak score within ±0.3s of a turn-end (pos) vs scores
        # ≥3s from any turn-end (neg) — the live analogue of s_pred AUC.
        pos = [s_arr[(t_arr >= t - 0.3) & (t_arr <= t + 0.3)].max()
               for t in in_rng if np.any((t_arr >= t - 0.3) & (t_arr <= t + 0.3))]
        far = np.ones(len(t_arr), bool)
        for t in gt:
            far &= np.abs(t_arr - t) > 3.0
        neg = list(s_arr[far])
        auc = float("nan")
        if pos and neg:
            from sklearn.metrics import roc_auc_score
            y = [1]*len(pos) + [0]*len(neg)
            auc = roc_auc_score(y, pos + neg)
        fire_frac = float((s_arr > thr_now).mean())
        return len(in_rng), caught, auc, fire_frac

    if len(gt):
        print(f"\n=== per-setback (thr={thr_now}, catch ±0.6s, AUC pos±0.3s vs neg>3s) ===")
        print(f"{'setback':>8} {'catch':>13} {'fire%':>7} {'sepAUC':>8}")
        for sb in SETBACKS_S:
            if not series[sb]["s"]:
                continue
            in_rng, caught, auc, firef = catch_and_auc(series[sb]["t"], series[sb]["s"])
            print(f"{sb:>6.1f}s  {caught:>3}/{in_rng:<3} ({100*caught/max(1,in_rng):>3.0f}%)"
                  f"  {100*firef:>5.1f}  {auc:>7.3f}")

    if out_png:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(len(SETBACKS_S), 1, figsize=(15, 9), sharex=True)
            for ax, sb in zip(axes, SETBACKS_S):
                ax.plot(series[sb]["t"], series[sb]["s"], lw=0.7)
                ax.axhline(thr_now, color="red", ls="--", lw=0.7)
                for t in gt:
                    ax.axvline(t, color="green", alpha=0.25, lw=0.7)
                ax.set_ylabel(f"setback {sb}s")
            axes[-1].set_xlabel("time (s)")
            axes[0].set_title("Live shift signal vs setback (green=GT human turn-ends)")
            plt.tight_layout(); plt.savefig(out_png, dpi=120)
            print(f"\nsaved plot: {out_png}")
        except Exception as e:
            print(f"  (plot skipped: {e})")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--audio",    default="processed_data/aeeee708.wav")
    ap.add_argument("--video",    default="raw_data/mediapipe_output/synced_video_60feat_stereo_bs.pkl")
    ap.add_argument("--textgrid", default="processed_data/aeeee708.TextGrid")
    ap.add_argument("--weights",  default=None, help="state_dict .pt (default: Candor pretrained)")
    ap.add_argument("--hop",      type=float, default=0.5)
    ap.add_argument("--window",   type=float, default=20.0)
    ap.add_argument("--thr",      type=float, default=0.58, help="EP0 s_pred_p_now sum-10 threshold")
    ap.add_argument("--fill-std", type=float, default=0.01, help="robot ch1 noise std (match fine-tune)")
    ap.add_argument("--out-png",  default="live_service/sim_signal.png")
    ap.add_argument("--max-seconds", type=float, default=None, help="cap sim length for a quick check")
    a = ap.parse_args()

    def _abs(p):
        p = Path(p); return p if p.is_absolute() else ROOT / p
    simulate(_abs(a.audio), _abs(a.video),
             _abs(a.textgrid) if a.textgrid else None,
             a.weights, a.hop, a.window, a.thr, a.fill_std, _abs(a.out_png),
             max_seconds=a.max_seconds)


if __name__ == "__main__":
    main()
