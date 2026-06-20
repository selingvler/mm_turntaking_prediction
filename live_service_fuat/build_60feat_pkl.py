#!/usr/bin/env python3
"""
Build the 60-feature pkl the turn-taking model expects, from the per-speaker
MediaPipe landmark pkls produced by mediapipe_feature_extraction.py.

Two modes (selected via CLI flag):
  python build_60feat_pkl.py
        Reads landmarks_left.pkl / landmarks_right.pkl, AU slots are zeros.
        Writes synced_video_60feat_{left,right,stereo}.pkl.

  python build_60feat_pkl.py --bs
        Reads landmarks_left_bs.pkl / landmarks_right_bs.pkl which contain
        52 blendshape scores per frame, and maps them to AU intensities via
        BLENDSHAPE_TO_AU. Writes synced_video_60feat_{left,right,stereo}_bs.pkl.

Layout of the 60 features (matches openface/convert_openface_to_pkl.py):
  [0:6]    gaze        — derived from iris vs eye-center
  [6:12]   pose        — Kabsch SVD on 6 canonical landmarks (pure numpy)
  [12:22]  jaw  (10)   — dlib points 3,6,8,10,13 (x,y) mapped to MediaPipe
  [22:26]  brow (4)    — dlib points 19,24       (x,y)
  [26:42]  nose (16)   — dlib points 27..34       (x,y)
  [42:59]  AU  (17)    — derived from MediaPipe ARKit blendshapes
                         (BLENDSHAPE_TO_AU below — best-effort FACS mapping;
                         libreface still skipped, but this is a real signal,
                         not zeros — works in dynamic real-time pipelines too)
  [59]     confidence  — face_detected (0/1)
"""

import argparse
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------
# MediaPipe -> dlib 68 mapping for the points OpenFace actually uses.
# Indices below are MediaPipe FaceMesh 478-landmark indices.
# --------------------------------------------------------------------------
DLIB_TO_MP = {
    # jaw line (dlib 3/6/8/10/13)
    3:  172,
    6:  136,
    8:  152,
    10: 365,
    13: 397,
    # eyebrows (dlib 19/24 — inner peaks of each brow)
    19: 105,
    24: 334,
    # nose bridge & tip (dlib 27..34)
    27: 168,
    28: 6,
    29: 197,
    30: 195,
    31: 5,
    32: 4,
    33: 1,
    34: 19,
}

NOSE_DLIB = list(range(27, 35))            # 8 pts
JAW_DLIB  = [3, 13, 6, 10, 8]               # 5 pts (order matches OF)
BROW_DLIB = [19, 24]                        # 2 pts


# Head-pose 3D model points (mm).
POSE_3D = np.array([
    (   0.0,    0.0,    0.0),    # nose tip
    (   0.0, -330.0,  -65.0),    # chin
    (-225.0,  170.0, -135.0),    # left  eye outer corner
    ( 225.0,  170.0, -135.0),    # right eye outer corner
    (-150.0, -150.0, -125.0),    # left  mouth corner
    ( 150.0, -150.0, -125.0),    # right mouth corner
], dtype=np.float64)
LMK_FOR_POSE = [1, 152, 33, 263, 61, 291]   # MediaPipe indices

# Gaze: iris centers vs eye centers (refine_landmarks=True gives 478 points).
EYE_RIGHT_CORNERS = (33, 133)
EYE_LEFT_CORNERS  = (362, 263)
IRIS_RIGHT_CENTER = 468
IRIS_LEFT_CENTER  = 473


# Final column order — must match callbacks.py / convert_openface_to_pkl.py.
GAZE_COLS = ["gaze_0_x", "gaze_0_y", "gaze_0_z",
             "gaze_1_x", "gaze_1_y", "gaze_1_z"]
POSE_COLS = ["pose_Tx", "pose_Ty", "pose_Tz",
             "pose_Rx", "pose_Ry", "pose_Rz"]
LMK_COLS  = (
    [f"x_{d}" for d in JAW_DLIB] + [f"y_{d}" for d in JAW_DLIB] +
    [f"x_{d}" for d in BROW_DLIB] + [f"y_{d}" for d in BROW_DLIB] +
    [f"x_{d}" for d in NOSE_DLIB] + [f"y_{d}" for d in NOSE_DLIB]
)
AU_COLS   = [f"AU{a:02d}_r" for a in
             (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20, 23, 25, 26, 45)]
CONF_COLS = ["confidence"]
FINAL_COLS = GAZE_COLS + POSE_COLS + LMK_COLS + AU_COLS + CONF_COLS  # 60


# --------------------------------------------------------------------------
# MediaPipe ARKit blendshape index reference (52 entries):
#   0: _neutral                  26: jawRight
#   1: browDownLeft              27: mouthClose
#   2: browDownRight             28: mouthDimpleLeft
#   3: browInnerUp               29: mouthDimpleRight
#   4: browOuterUpLeft           30: mouthFrownLeft
#   5: browOuterUpRight          31: mouthFrownRight
#   6: cheekPuff                 32: mouthFunnel
#   7: cheekSquintLeft           33: mouthLeft
#   8: cheekSquintRight          34: mouthLowerDownLeft
#   9: eyeBlinkLeft              35: mouthLowerDownRight
#  10: eyeBlinkRight             36: mouthPressLeft
#  11: eyeLookDownLeft           37: mouthPressRight
#  12: eyeLookDownRight          38: mouthPucker
#  13: eyeLookInLeft             39: mouthRight
#  14: eyeLookInRight            40: mouthRollLower
#  15: eyeLookOutLeft            41: mouthRollUpper
#  16: eyeLookOutRight           42: mouthShrugLower
#  17: eyeLookUpLeft             43: mouthShrugUpper
#  18: eyeLookUpRight            44: mouthSmileLeft
#  19: eyeSquintLeft             45: mouthSmileRight
#  20: eyeSquintRight            46: mouthStretchLeft
#  21: eyeWideLeft               47: mouthStretchRight
#  22: eyeWideRight              48: mouthUpperUpLeft
#  23: jawForward                49: mouthUpperUpRight
#  24: jawLeft                   50: noseSneerLeft
#  25: jawOpen                   51: noseSneerRight
#
# AU -> list of blendshape indices (averaged). Best-effort FACS mapping.
# Order MUST match AU_COLS above (1, 2, 4, 5, 6, 7, 9, 10, 12, 14, 15, 17, 20,
# 23, 25, 26, 45).
BLENDSHAPE_TO_AU = [
    [3],          # AU01  inner brow raise   browInnerUp
    [4, 5],       # AU02  outer brow raise   browOuterUpLeft + Right
    [1, 2],       # AU04  brow lower         browDownLeft + Right
    [21, 22],     # AU05  upper lid raise    eyeWideLeft + Right
    [7, 8],       # AU06  cheek raise        cheekSquintLeft + Right
    [19, 20],     # AU07  lid tighten        eyeSquintLeft + Right
    [50, 51],     # AU09  nose wrinkle       noseSneerLeft + Right
    [48, 49],     # AU10  upper lip raise    mouthUpperUpLeft + Right
    [44, 45],     # AU12  lip corner pull    mouthSmileLeft + Right
    [28, 29],     # AU14  dimpler            mouthDimpleLeft + Right
    [30, 31],     # AU15  lip corner depress mouthFrownLeft + Right
    [43],         # AU17  chin raise         mouthShrugUpper
    [46, 47],     # AU20  lip stretch        mouthStretchLeft + Right
    [36, 37],     # AU23  lip tighten        mouthPressLeft + Right
    [25],         # AU25  lips part          jawOpen (partial)
    [25],         # AU26  jaw drop           jawOpen
    [9, 10],      # AU45  blink              eyeBlinkLeft + Right
]
NUM_BLENDSHAPES = 52


def landmarks_array(df: pd.DataFrame) -> np.ndarray:
    """(T, 478, 3) float32 from a wide df with x_i/y_i/z_i columns."""
    n = 478
    xs = df[[f"x_{i}" for i in range(n)]].to_numpy(np.float32)
    ys = df[[f"y_{i}" for i in range(n)]].to_numpy(np.float32)
    zs = df[[f"z_{i}" for i in range(n)]].to_numpy(np.float32)
    return np.stack([xs, ys, zs], axis=-1)


def derive_gaze(lm: np.ndarray) -> np.ndarray:
    """(T, 6): right then left (matches OpenFace gaze_0 / gaze_1)."""
    eye_r = (lm[:, EYE_RIGHT_CORNERS[0]] + lm[:, EYE_RIGHT_CORNERS[1]]) / 2.0
    eye_l = (lm[:, EYE_LEFT_CORNERS[0]]  + lm[:, EYE_LEFT_CORNERS[1]])  / 2.0
    iris_r = lm[:, IRIS_RIGHT_CENTER]
    iris_l = lm[:, IRIS_LEFT_CENTER]
    g_r = iris_r - eye_r
    g_l = iris_l - eye_l
    return np.concatenate([g_r, g_l], axis=1).astype(np.float32)


def _rotation_to_rodrigues(R: np.ndarray) -> np.ndarray:
    cos_t = (np.trace(R) - 1.0) * 0.5
    cos_t = np.clip(cos_t, -1.0, 1.0)
    theta = np.arccos(cos_t)
    if theta < 1e-6:
        return np.zeros(3, dtype=np.float64)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=np.float64) / (2.0 * np.sin(theta))
    return axis * theta


def derive_pose(lm: np.ndarray) -> np.ndarray:
    """(T, 6) Tx/Ty/Tz, Rx/Ry/Rz via Kabsch alignment of 3D landmarks
    against the canonical 3D face model. Pure numpy, no cv2."""
    T = lm.shape[0]
    out = np.zeros((T, 6), dtype=np.float32)

    canonical = POSE_3D.astype(np.float64)
    c_mean = canonical.mean(axis=0)
    canonical_c = canonical - c_mean

    for t in range(T):
        obs = lm[t, LMK_FOR_POSE].astype(np.float64)
        o_mean = obs.mean(axis=0)
        obs_c = obs - o_mean

        H = canonical_c.T @ obs_c
        U, _, Vt = np.linalg.svd(H)
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        D = np.diag([1.0, 1.0, d])
        R = Vt.T @ D @ U.T

        out[t, 0:3] = o_mean
        out[t, 3:6] = _rotation_to_rodrigues(R)
    return out


def derive_lmk_subset(lm: np.ndarray) -> pd.DataFrame:
    cols = {}
    for d in JAW_DLIB:
        cols[f"x_{d}"] = lm[:, DLIB_TO_MP[d], 0]
    for d in JAW_DLIB:
        cols[f"y_{d}"] = lm[:, DLIB_TO_MP[d], 1]
    for d in BROW_DLIB:
        cols[f"x_{d}"] = lm[:, DLIB_TO_MP[d], 0]
    for d in BROW_DLIB:
        cols[f"y_{d}"] = lm[:, DLIB_TO_MP[d], 1]
    for d in NOSE_DLIB:
        cols[f"x_{d}"] = lm[:, DLIB_TO_MP[d], 0]
    for d in NOSE_DLIB:
        cols[f"y_{d}"] = lm[:, DLIB_TO_MP[d], 1]
    return pd.DataFrame(cols, dtype=np.float32)


def normalize_inplace(df: pd.DataFrame, cols):
    """OpenFace-style: per-file min-max then mean-center."""
    sub = df[cols].astype(np.float64)
    mn = sub.min()
    mx = sub.max()
    rng = (mx - mn).replace(0, 1.0)
    sub = (sub - mn) / rng
    sub = sub - sub.mean()
    df[cols] = sub.astype(np.float32).values


def derive_aus_from_blendshapes(df_in: pd.DataFrame) -> np.ndarray:
    """Average the assigned blendshape scores per AU. Returns (T, 17)."""
    bs_cols = [f"bs_{i}" for i in range(NUM_BLENDSHAPES)]
    if not all(c in df_in.columns for c in bs_cols):
        sys.exit("input pkl has no blendshape columns; rerun extraction with --bs")
    bs = df_in[bs_cols].to_numpy(np.float32)            # (T, 52)
    aus = np.zeros((len(df_in), len(AU_COLS)), dtype=np.float32)
    for i, src_indices in enumerate(BLENDSHAPE_TO_AU):
        aus[:, i] = bs[:, src_indices].mean(axis=1)
    return aus


def build_one_speaker(df_in: pd.DataFrame, label: str, use_blendshapes: bool) -> pd.DataFrame:
    print(f"[{label}] landmarks df: {df_in.shape}", flush=True)
    lm = landmarks_array(df_in)
    print(f"[{label}] landmark tensor: {lm.shape}", flush=True)

    gaze = derive_gaze(lm)
    df_gaze = pd.DataFrame(gaze, columns=GAZE_COLS)

    pose = derive_pose(lm)
    df_pose = pd.DataFrame(pose, columns=POSE_COLS)

    df_lmk = derive_lmk_subset(lm)

    if use_blendshapes:
        aus = derive_aus_from_blendshapes(df_in)
        nonzero_frac = float((aus.any(axis=1)).mean())
        print(f"[{label}] AU source: blendshape mapping  "
              f"(non-zero fraction {nonzero_frac:.3f})", flush=True)
    else:
        aus = np.zeros((len(df_in), len(AU_COLS)), dtype=np.float32)
        print(f"[{label}] AU source: zeros (default mode)", flush=True)
    df_au = pd.DataFrame(aus, columns=AU_COLS)

    df_conf = pd.DataFrame({
        "confidence": df_in["face_detected"].astype(np.float32).values
    })

    df = pd.concat([df_gaze, df_pose, df_lmk, df_au, df_conf], axis=1)
    df = df[FINAL_COLS]   # enforce column order
    print(f"[{label}] combined: {df.shape}  (expected: ({len(df_in)}, 60))", flush=True)

    normalize_inplace(df, GAZE_COLS + POSE_COLS + list(df_lmk.columns))
    return df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--bs", action="store_true",
        help="Use blendshape-derived AU intensities (reads *_bs.pkl, writes *_bs.pkl). "
             "Default: AU slots are zeros, *non-bs* file names.",
    )
    args = parser.parse_args()

    suffix = "_bs" if args.bs else ""
    # TAG env-var picks a recording-specific naming pattern that matches
    # mediapipe_feature_extraction.py's output (landmarks_{TAG}_{left,right}{_bs}).
    tag = os.environ.get("TAG", "")
    tag_part = f"{tag}_" if tag else ""
    in_dir  = Path("raw_data/mediapipe_output")
    in_left  = in_dir / f"landmarks_{tag_part}left{suffix}.pkl"
    in_right = in_dir / f"landmarks_{tag_part}right{suffix}.pkl"
    out_dir  = in_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    out_left   = out_dir / f"synced_video_60feat_{tag_part}left{suffix}.pkl"
    out_right  = out_dir / f"synced_video_60feat_{tag_part}right{suffix}.pkl"
    out_stereo = out_dir / f"synced_video_60feat_{tag_part}stereo{suffix}.pkl"

    if not in_left.exists() or not in_right.exists():
        sys.exit(f"need both pkls. left={in_left.exists()} right={in_right.exists()}")

    with open(in_left, "rb")  as f: dl = pickle.load(f)
    with open(in_right, "rb") as f: dr = pickle.load(f)
    fps = dl["fps"]
    print(f"fps={fps}, mode={'blendshape AU' if args.bs else 'AU=0'}", flush=True)

    df_l = build_one_speaker(dl["df"], "left",  use_blendshapes=args.bs)
    df_r = build_one_speaker(dr["df"], "right", use_blendshapes=args.bs)

    if len(df_l) != len(df_r):
        sys.exit(f"frame count mismatch: left={len(df_l)} right={len(df_r)}")

    with open(out_left, "wb")  as f: pickle.dump(df_l, f)
    with open(out_right, "wb") as f: pickle.dump(df_r, f)
    print(f"saved per-speaker: {out_left.name} / {out_right.name}", flush=True)

    arr = np.stack(
        [df_l[FINAL_COLS].to_numpy(np.float32),
         df_r[FINAL_COLS].to_numpy(np.float32)],
        axis=-1,
    )  # (T, 60, 2)
    stereo = {
        "arr": arr,
        "columns": FINAL_COLS,
        "fps": fps,
    }
    with open(out_stereo, "wb") as f:
        pickle.dump(stereo, f)
    print(f"saved stereo: {out_stereo} shape={arr.shape}", flush=True)
    print(f"NaN cells (left + right): "
          f"{int(df_l.isna().sum().sum() + df_r.isna().sum().sum())}", flush=True)


if __name__ == "__main__":
    main()
