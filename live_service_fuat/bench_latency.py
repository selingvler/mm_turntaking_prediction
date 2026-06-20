#!/usr/bin/env python3
"""
Gate #2: forward-pass latency benchmark for the live service.

Answers the only question that decides whether the multimodal live path is
viable: how long does ONE EarlyVAFusion forward pass take on this machine, at a
given window length, on CPU / MPS?

For real-time we need:   forward_pass_wall_time + HOP_S  <  acceptable latency
and ideally                forward_pass_wall_time        <  HOP_S
so the predictor keeps up without falling behind the audio stream.

Usage:
    python live_service/bench_latency.py                 # sweep 4,6,8,12,20 s
    WINDOW_S=8 python live_service/bench_latency.py       # single window
    N_ITERS=50 python live_service/bench_latency.py

No real audio/video needed — random tensors of the right shape are enough to
time the math. We reuse run_early_fusion.py's model loader (and its CUDA->CPU
monkey-patches) so the architecture/weights are identical to inference.
"""

import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "selinc"))

# run_early_fusion installs the CUDA->CPU/MPS patches at import and exposes
# load_model() + _TARGET_DEVICE. Importing it is enough to get a ready model.
import run_early_fusion as ref  # noqa: E402


AUDIO_SR   = 16_000
VIDEO_FPS  = 30
WINDOWS_S  = ([float(os.environ["WINDOW_S"])] if os.environ.get("WINDOW_S")
              else [4.0, 6.0, 8.0, 12.0, 20.0])
N_ITERS    = int(os.environ.get("N_ITERS", "20"))
N_WARMUP   = int(os.environ.get("N_WARMUP", "3"))


def make_window(window_s: float, device) -> dict:
    """Random stereo audio + 60-dim stereo video of the right length.
    audio: (1, window_s*16000, 2)   video: (1, window_s*30, 60, 2)
    Length rule from DESIGN §0: audio@16k + video@30fps keeps the upsampler
    aligned (audio->W*50 emb frames, video W*30 ->x1.667-> W*50)."""
    a_len = int(round(window_s * AUDIO_SR))
    v_len = int(round(window_s * VIDEO_FPS))
    audio = torch.randn(1, a_len, 2, device=device)
    frames = torch.randn(1, v_len, 60, 2, device=device)
    return {"audio_chunk": audio, "frames": frames}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def bench_one(model, device, window_s: float) -> dict:
    batch_factory = lambda: make_window(window_s, device)

    with torch.no_grad():
        for _ in range(N_WARMUP):                      # warmup / lazy init
            model(batch_factory())
        sync(device)

        times = []
        for _ in range(N_ITERS):
            batch = batch_factory()
            sync(device)
            t0 = time.perf_counter()
            model(batch)
            sync(device)
            times.append(time.perf_counter() - t0)

    times = np.array(times)
    return {
        "window_s": window_s,
        "mean_ms":  float(times.mean() * 1e3),
        "p50_ms":   float(np.percentile(times, 50) * 1e3),
        "p95_ms":   float(np.percentile(times, 95) * 1e3),
        "max_ms":   float(times.max() * 1e3),
    }


def main():
    device = ref._TARGET_DEVICE
    print(f"device      : {device}")
    print(f"iters       : {N_ITERS} (+{N_WARMUP} warmup)")
    print(f"windows (s) : {WINDOWS_S}")
    print("loading model (EarlyVAFusion, candor weights)...")
    model = ref.load_model()

    print(f"\n{'window':>8} {'mean':>9} {'p50':>9} {'p95':>9} {'max':>9}   verdict")
    print("-" * 64)
    for w in WINDOWS_S:
        r = bench_one(model, device, w)
        # "real-time" heuristic: a pass must finish well under a 0.5s hop to
        # keep up. Flag green if mean < 250ms, amber < 500ms, else red.
        v = ("OK (<0.25s)" if r["mean_ms"] < 250 else
             "tight (<0.5s)" if r["mean_ms"] < 500 else
             "TOO SLOW (>0.5s)")
        print(f"{w:>6.0f}s  {r['mean_ms']:>7.1f}ms {r['p50_ms']:>7.1f}ms "
              f"{r['p95_ms']:>7.1f}ms {r['max_ms']:>7.1f}ms   {v}")

    print("\nInterpretation:")
    print("  Live predictor re-runs every HOP_S (default 0.5s). The chosen")
    print("  WINDOW_S is viable if its mean pass time stays comfortably below")
    print("  HOP_S, leaving headroom for capture + STT on the same machine.")
    print("  If the smallest acceptable window (>=6s, DESIGN §2) is still red,")
    print("  fall back to the audio-only model (fuat_scripts/tahmin_fuat.py).")


if __name__ == "__main__":
    main()
