#!/usr/bin/env python3
"""
MediaPipe-based facial landmark extraction for a Teams-style split-screen video.

Pipeline:
  - Skips the first SKIP_SECONDS of video (intro / setup).
  - For every remaining frame, splits the frame vertically in half:
        left  half -> speaker 0
        right half -> speaker 1
    and runs MediaPipe FaceLandmarker independently on each half.
  - Captures both 478 3D landmarks AND 52 ARKit-style face blendshapes per face.
        (Blendshapes are used downstream by build_60feat_pkl.py to fill the AU
         slots of the model's 60-feature vector — replacing the previous AU=0.)
  - Adds timestamp = frame_idx / fps.
  - Hybrid fill for missing detections:
        zeros until first hit, then carry last valid frame forward.
  - Builds [T, D] feature matrix per speaker
        (478 landmarks * 3 = 1434 dims + 52 blendshape scores + meta cols).
  - Saves two pkls (left / right) + a small visualization comparing both speakers.
"""

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import pickle
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt


VIDEO_PATH = Path("buse_workspace/session02/raw/video.mp4")
MODEL_PATH = Path("buse_workspace/models/face_landmarker.task")
OUTPUT_DIR = Path("buse_workspace/session02/mediapipe_output")
# "_bs" suffix = this run captures blendshapes alongside landmarks.
# The previous landmarks_left.pkl / landmarks_right.pkl (no blendshapes) are
# left untouched so you can compare AU=0 vs blendshape-derived AU downstream.
OUT_LEFT_PKL  = OUTPUT_DIR / "landmarks_left_bs.pkl"
OUT_RIGHT_PKL = OUTPUT_DIR / "landmarks_right_bs.pkl"
PLOT_FILE     = OUTPUT_DIR / "landmark_analysis_bs.png"

SKIP_SECONDS    = 0        # drop first minute (intro / setup)
NUM_LANDMARKS   = 478
NUM_BLENDSHAPES = 52          # ARKit-style blendshape vector returned by MediaPipe


def make_landmarker():
    base_opts = mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH))
    opts = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=base_opts,
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
        min_face_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return mp.tasks.vision.FaceLandmarker.create_from_options(opts)


def detect_one(landmarker, half_bgr, ts_ms):
    rgb = cv2.cvtColor(half_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def landmarks_to_vec(result, dim):
    if not result.face_landmarks:
        return None
    lm = result.face_landmarks[0]
    return np.fromiter(
        (c for p in lm for c in (p.x, p.y, p.z)),
        dtype=np.float32, count=dim,
    )


def blendshapes_to_vec(result, dim):
    if not result.face_blendshapes:
        return None
    bs = result.face_blendshapes[0]
    return np.fromiter((c.score for c in bs),
                       dtype=np.float32, count=dim)


def to_dataframe(feats: np.ndarray, blendshapes: np.ndarray,
                 detected: np.ndarray, timestamps: np.ndarray) -> pd.DataFrame:
    cols = []
    for i in range(NUM_LANDMARKS):
        cols += [f"x_{i}", f"y_{i}", f"z_{i}"]
    df = pd.DataFrame(feats, columns=cols)
    bs_cols = [f"bs_{i}" for i in range(NUM_BLENDSHAPES)]
    df_bs = pd.DataFrame(blendshapes, columns=bs_cols)
    df = pd.concat([df, df_bs], axis=1)
    df.insert(0, "face_detected", detected)
    df.insert(0, "timestamp", timestamps)
    df.insert(0, "frame", np.arange(len(df), dtype=np.int64))
    return df


def extract():
    if not VIDEO_PATH.exists():
        sys.exit(f"video not found: {VIDEO_PATH}")
    if not MODEL_PATH.exists():
        sys.exit(f"model not found: {MODEL_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))
    if not cap.isOpened():
        sys.exit(f"could not open video: {VIDEO_PATH}")
    fps    = cap.get(cv2.CAP_PROP_FPS)
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    half_w = width // 2
    skip_frames = int(round(SKIP_SECONDS * fps))
    process_total = max(0, total - skip_frames)
    print(f"video: fps={fps}, total={total}, size={width}x{height}, "
          f"skip_first_frames={skip_frames}, process={process_total}", flush=True)

    landmarker_l = make_landmarker()
    landmarker_r = make_landmarker()

    feat_dim = NUM_LANDMARKS * 3
    feats_l = np.zeros((process_total, feat_dim), dtype=np.float32)
    feats_r = np.zeros((process_total, feat_dim), dtype=np.float32)
    bs_l    = np.zeros((process_total, NUM_BLENDSHAPES), dtype=np.float32)
    bs_r    = np.zeros((process_total, NUM_BLENDSHAPES), dtype=np.float32)
    det_l   = np.zeros(process_total, dtype=np.int8)
    det_r   = np.zeros(process_total, dtype=np.int8)
    last_l       = None
    last_r       = None
    last_bs_l    = None
    last_bs_r    = None

    # advance over skipped frames cheaply
    cap.set(cv2.CAP_PROP_POS_FRAMES, skip_frames)
    out_idx = 0
    t0 = time.time()
    src_idx = skip_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if out_idx >= process_total:
            break

        ts_ms = int(round(1000.0 * src_idx / fps))
        left_half  = frame[:, :half_w]
        right_half = frame[:, half_w:]

        r_l = detect_one(landmarker_l, left_half,  ts_ms)
        r_r = detect_one(landmarker_r, right_half, ts_ms)

        v_l = landmarks_to_vec(r_l, feat_dim)
        v_r = landmarks_to_vec(r_r, feat_dim)
        b_l = blendshapes_to_vec(r_l, NUM_BLENDSHAPES)
        b_r = blendshapes_to_vec(r_r, NUM_BLENDSHAPES)

        if v_l is not None:
            feats_l[out_idx] = v_l
            det_l[out_idx]   = 1
            last_l = v_l
        elif last_l is not None:
            feats_l[out_idx] = last_l
        # else stays zero

        if b_l is not None:
            bs_l[out_idx] = b_l
            last_bs_l = b_l
        elif last_bs_l is not None:
            bs_l[out_idx] = last_bs_l

        if v_r is not None:
            feats_r[out_idx] = v_r
            det_r[out_idx]   = 1
            last_r = v_r
        elif last_r is not None:
            feats_r[out_idx] = last_r

        if b_r is not None:
            bs_r[out_idx] = b_r
            last_bs_r = b_r
        elif last_bs_r is not None:
            bs_r[out_idx] = last_bs_r

        out_idx += 1
        src_idx += 1
        if out_idx % 1000 == 0:
            elapsed = time.time() - t0
            rate = out_idx / elapsed if elapsed > 0 else 0
            eta = (process_total - out_idx) / rate if rate > 0 else 0
            print(f"  {out_idx}/{process_total}  rate={rate:.1f} fps  ETA={eta/60:.1f} min  "
                  f"L_det={det_l[:out_idx].mean():.2f}  R_det={det_r[:out_idx].mean():.2f}",
                  flush=True)

    cap.release()
    landmarker_l.close()
    landmarker_r.close()

    feats_l = feats_l[:out_idx]
    feats_r = feats_r[:out_idx]
    bs_l    = bs_l[:out_idx]
    bs_r    = bs_r[:out_idx]
    det_l   = det_l[:out_idx]
    det_r   = det_r[:out_idx]
    timestamps = (np.arange(out_idx, dtype=np.float64) + skip_frames) / fps

    print(f"\nfinal frames: {out_idx}", flush=True)
    print(f"  left  detected: {int(det_l.sum())}/{out_idx} "
          f"({100*det_l.mean():.1f}%)", flush=True)
    print(f"  right detected: {int(det_r.sum())}/{out_idx} "
          f"({100*det_r.mean():.1f}%)", flush=True)

    df_l = to_dataframe(feats_l, bs_l, det_l, timestamps)
    df_r = to_dataframe(feats_r, bs_r, det_r, timestamps)
    return df_l, df_r, fps


def visualize(df_l, df_r, fps, plot_path):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    fig.suptitle("MediaPipe stereo extraction — quality overview",
                 fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(df_l["timestamp"], df_l["face_detected"],
            color="steelblue", linewidth=0.6, label="left (speaker 0)")
    ax.plot(df_r["timestamp"], df_r["face_detected"] - 1.1,
            color="firebrick", linewidth=0.6, label="right (speaker 1)")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("detected (per speaker, offset)")
    ax.set_title("Detection over time")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    nose_l = df_l[df_l["face_detected"] == 1]
    nose_r = df_r[df_r["face_detected"] == 1]
    if len(nose_l):
        ax.plot(nose_l["timestamp"], nose_l["x_1"],
                color="steelblue", linewidth=0.5, label="left x")
    if len(nose_r):
        ax.plot(nose_r["timestamp"], nose_r["x_1"],
                color="firebrick", linewidth=0.5, label="right x")
    ax.set_xlabel("time (s)")
    ax.set_ylabel("nose-tip x (normalized)")
    ax.set_title("Nose-tip horizontal trajectory")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    runs_l = (df_l["face_detected"] != df_l["face_detected"].shift()).cumsum()
    miss_l = df_l.loc[df_l["face_detected"] == 0].groupby(runs_l).size()
    runs_r = (df_r["face_detected"] != df_r["face_detected"].shift()).cumsum()
    miss_r = df_r.loc[df_r["face_detected"] == 0].groupby(runs_r).size()
    bins = 30
    if len(miss_l):
        ax.hist(miss_l.values, bins=bins, alpha=0.6,
                color="steelblue", edgecolor="black", label="left")
    if len(miss_r):
        ax.hist(miss_r.values, bins=bins, alpha=0.6,
                color="firebrick", edgecolor="black", label="right")
    ax.set_yscale("log")
    ax.set_xlabel("missing-face run length (frames)")
    ax.set_ylabel("count (log)")
    ax.set_title("Missing-face gap distribution")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    txt = (
        f"frames        : {len(df_l)} (per speaker)\n"
        f"fps           : {fps}\n"
        f"duration      : {df_l['timestamp'].iloc[-1] - df_l['timestamp'].iloc[0]:.1f} s\n"
        f"               (start at {df_l['timestamp'].iloc[0]:.1f} s, skip = {SKIP_SECONDS}s)\n"
        f"left  det     : {int(df_l['face_detected'].sum())} "
        f"({100*df_l['face_detected'].mean():.1f}%)\n"
        f"right det     : {int(df_r['face_detected'].sum())} "
        f"({100*df_r['face_detected'].mean():.1f}%)\n"
        f"feature dim   : 1434 (xyz × 478)\n"
    )
    ax.text(0.05, 0.5, txt, fontsize=11, family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.savefig(plot_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"saved plot: {plot_path}", flush=True)


def main():
    df_l, df_r, fps = extract()

    with open(OUT_LEFT_PKL, "wb") as f:
        pickle.dump({"df": df_l, "fps": fps}, f)
    with open(OUT_RIGHT_PKL, "wb") as f:
        pickle.dump({"df": df_r, "fps": fps}, f)
    print(f"saved: {OUT_LEFT_PKL}  ({OUT_LEFT_PKL.stat().st_size / 1e6:.1f} MB)", flush=True)
    print(f"saved: {OUT_RIGHT_PKL} ({OUT_RIGHT_PKL.stat().st_size / 1e6:.1f} MB)", flush=True)

    visualize(df_l, df_r, fps, PLOT_FILE)
    print("done.", flush=True)


if __name__ == "__main__":
    main()
