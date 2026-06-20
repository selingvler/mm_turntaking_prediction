#!/usr/bin/env python3
"""
Live per-frame 60-dim face features for the streaming predictor.

Runs MediaPipe FaceLandmarker (with blendshapes) on one webcam frame and builds
the SAME 60-feature vector as selinc/build_60feat_pkl.py by reusing its exact
derivations (gaze / pose / landmark-subset / AU-from-blendshapes), so live video
matches the training/offline pipeline.

KNOWN PARITY CAVEAT
-------------------
build_60feat_pkl normalizes some columns per *session* (normalize_inplace over
the whole recording). Streaming has no full session, so this skips that step
(raw features). If live video quality is poor, this is the first thing to fix —
maintain running mean/var per feature and normalize online, or z-score against
stats collected during a short calibration window.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "selinc"))
import build_60feat_pkl as B           # reuse the exact offline derivations  # noqa: E402

_MODEL_CANDIDATES = [
    ROOT / "models" / "face_landmarker.task",
    ROOT / "face_landmarker.task",
]


class LiveFaceFeatures:
    def __init__(self, model_path: str | None = None, fps: float = 30.0):
        import mediapipe as mp
        self.mp = mp
        path = Path(model_path) if model_path else next(
            (p for p in _MODEL_CANDIDATES if p.exists()), None)
        if path is None or not path.exists():
            raise FileNotFoundError(
                f"face_landmarker.task not found in {[str(p) for p in _MODEL_CANDIDATES]}")
        opts = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=str(path)),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1, output_face_blendshapes=True,
            min_face_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.lm = mp.tasks.vision.FaceLandmarker.create_from_options(opts)
        self.dt_ms = int(round(1000.0 / fps))
        self._ts = 0
        self._last = np.zeros(60, np.float32)   # carry-forward on miss (offline behaviour)
        # online version of build_60feat.normalize_inplace (per-file min-max then
        # mean-center) over the first 42 dims (gaze+pose+lmk). Running min/max/mean
        # from service start; net transform = (x - mean) / (max - min).
        self._NN = 42
        self._mn = self._mx = self._sum = None
        self._cnt = 0

    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        x = vec[:self._NN]
        if self._mn is None:
            self._mn = x.copy(); self._mx = x.copy()
            self._sum = x.astype(np.float64).copy(); self._cnt = 1
        else:
            np.minimum(self._mn, x, out=self._mn)
            np.maximum(self._mx, x, out=self._mx)
            self._sum += x; self._cnt += 1
        mean = (self._sum / self._cnt).astype(np.float32)
        rng = (self._mx - self._mn).astype(np.float32)
        rng[rng < 1e-6] = 1.0
        vec[:self._NN] = (x - mean) / rng
        return vec

    def frame_to_60(self, frame_bgr) -> np.ndarray:
        import cv2
        self._ts += self.dt_ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=rgb)
        res = self.lm.detect_for_video(img, self._ts)
        if not res.face_landmarks:
            return self._last
        pts = res.face_landmarks[0]
        row = {}
        for i, p in enumerate(pts):
            row[f"x_{i}"] = p.x; row[f"y_{i}"] = p.y; row[f"z_{i}"] = p.z
        if res.face_blendshapes:
            for i, b in enumerate(res.face_blendshapes[0]):
                row[f"bs_{i}"] = b.score
        df = pd.DataFrame([row])

        lm = B.landmarks_array(df)                       # (1,478,3)
        gaze = B.derive_gaze(lm)                         # (1,6)
        pose = B.derive_pose(lm)                         # (1,6)
        lmk = B.derive_lmk_subset(lm).values             # (1,30)
        aus = B.derive_aus_from_blendshapes(df)          # (1,17)
        conf = np.ones((1, 1), np.float32)               # detected
        vec = np.concatenate([gaze, pose, lmk, aus, conf], axis=1)[0].astype(np.float32)
        vec = self._normalize(vec)                        # online min-max + mean-center
        self._last = vec
        return self._last
