import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import torch
from pathlib import Path

MODEL_PATH = Path("buse_workspace/models/face_landmarker.task")

NUM_LANDMARKS = 478
NUM_BLENDSHAPES = 52

DLIB_TO_MP = {
    3: 172,
    6: 136,
    8: 152,
    10: 365,
    13: 397,
    19: 105,
    24: 334,
    27: 168,
    28: 6,
    29: 197,
    30: 195,
    31: 5,
    32: 4,
    33: 1,
    34: 19,
}

NOSE_DLIB = list(range(27, 35))
JAW_DLIB = [3, 13, 6, 10, 8]
BROW_DLIB = [19, 24]

POSE_3D = np.array([
    (0.0, 0.0, 0.0),
    (0.0, -330.0, -65.0),
    (-225.0, 170.0, -135.0),
    (225.0, 170.0, -135.0),
    (-150.0, -150.0, -125.0),
    (150.0, -150.0, -125.0),
], dtype=np.float64)

LMK_FOR_POSE = [1, 152, 33, 263, 61, 291]

EYE_RIGHT_CORNERS = (33, 133)
EYE_LEFT_CORNERS = (362, 263)
IRIS_RIGHT_CENTER = 468
IRIS_LEFT_CENTER = 473

BLENDSHAPE_TO_AU = [
    [3],
    [4, 5],
    [1, 2],
    [21, 22],
    [7, 8],
    [19, 20],
    [50, 51],
    [48, 49],
    [44, 45],
    [28, 29],
    [30, 31],
    [43],
    [46, 47],
    [36, 37],
    [25],
    [25],
    [9, 10],
]


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


def detect_one(landmarker, frame_bgr, ts_ms):
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    return landmarker.detect_for_video(mp_image, ts_ms)


def landmarks_to_array(result):
    if not result.face_landmarks:
        return None

    lm = result.face_landmarks[0]

    arr = np.zeros((NUM_LANDMARKS, 3), dtype=np.float32)

    for i, p in enumerate(lm):
        if i >= NUM_LANDMARKS:
            break
        arr[i] = [p.x, p.y, p.z]

    return arr


def blendshapes_to_array(result):
    if not result.face_blendshapes:
        return np.zeros(NUM_BLENDSHAPES, dtype=np.float32)

    bs = result.face_blendshapes[0]
    arr = np.zeros(NUM_BLENDSHAPES, dtype=np.float32)

    for i, c in enumerate(bs):
        if i >= NUM_BLENDSHAPES:
            break
        arr[i] = c.score

    return arr


def derive_gaze(lm):
    eye_r = (lm[EYE_RIGHT_CORNERS[0]] + lm[EYE_RIGHT_CORNERS[1]]) / 2.0
    eye_l = (lm[EYE_LEFT_CORNERS[0]] + lm[EYE_LEFT_CORNERS[1]]) / 2.0
    iris_r = lm[IRIS_RIGHT_CENTER]
    iris_l = lm[IRIS_LEFT_CENTER]

    g_r = iris_r - eye_r
    g_l = iris_l - eye_l

    return np.concatenate([g_r, g_l]).astype(np.float32)


def _rotation_to_rodrigues(R):
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


def derive_pose(lm):
    canonical = POSE_3D.astype(np.float64)
    c_mean = canonical.mean(axis=0)
    canonical_c = canonical - c_mean

    obs = lm[LMK_FOR_POSE].astype(np.float64)
    o_mean = obs.mean(axis=0)
    obs_c = obs - o_mean

    H = canonical_c.T @ obs_c
    U, _, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    out = np.zeros(6, dtype=np.float32)
    out[0:3] = o_mean
    out[3:6] = _rotation_to_rodrigues(R)

    return out


def derive_lmk_subset(lm):
    values = []

    for d in JAW_DLIB:
        values.append(lm[DLIB_TO_MP[d], 0])
    for d in JAW_DLIB:
        values.append(lm[DLIB_TO_MP[d], 1])

    for d in BROW_DLIB:
        values.append(lm[DLIB_TO_MP[d], 0])
    for d in BROW_DLIB:
        values.append(lm[DLIB_TO_MP[d], 1])

    for d in NOSE_DLIB:
        values.append(lm[DLIB_TO_MP[d], 0])
    for d in NOSE_DLIB:
        values.append(lm[DLIB_TO_MP[d], 1])

    return np.array(values, dtype=np.float32)


def derive_aus(bs):
    aus = np.zeros(17, dtype=np.float32)

    for i, src_indices in enumerate(BLENDSHAPE_TO_AU):
        aus[i] = bs[src_indices].mean()

    return aus


def build_60_feature(lm, bs, detected):
    if lm is None:
        return np.zeros(60, dtype=np.float32)

    gaze = derive_gaze(lm)
    pose = derive_pose(lm)
    lmk = derive_lmk_subset(lm)
    aus = derive_aus(bs)
    confidence = np.array([1.0 if detected else 0.0], dtype=np.float32)

    feat = np.concatenate([
        gaze,
        pose,
        lmk,
        aus,
        confidence,
    ]).astype(np.float32)

    return feat


class LiveVideoExtractor:
    def __init__(
        self,
        fps=16,
        target_len=600,
        camera_index=0,
    ):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"MediaPipe model not found: {MODEL_PATH}")

        self.fps = fps
        self.target_len = target_len
        self.camera_index = camera_index

        self.landmarker = make_landmarker()

        self.cap = cv2.VideoCapture(camera_index)

        if not self.cap.isOpened():
            raise RuntimeError("Webcam açılamadı.")

        self.buffer = np.zeros(
            (target_len, 60, 2),
            dtype=np.float32,
        )

        self.frame_idx = 0
        self.last_feat = np.zeros(60, dtype=np.float32)

    def read_one_feature(self):
        ret, frame = self.cap.read()

        if not ret:
            return self.last_feat

        ts_ms = int(1000.0 * self.frame_idx / self.fps)
        self.frame_idx += 1

        result = detect_one(
            self.landmarker,
            frame,
            ts_ms,
        )

        lm = landmarks_to_array(result)
        bs = blendshapes_to_array(result)

        detected = lm is not None

        feat = build_60_feature(
            lm,
            bs,
            detected,
        )

        if detected:
            self.last_feat = feat

        return feat

    def update(self):
        feat_human = self.read_one_feature()

        feat_robot = np.zeros(
            60,
            dtype=np.float32,
        )

        stereo_feat = np.stack(
            [feat_human, feat_robot],
            axis=-1,
        )

        self.buffer = np.concatenate(
            [
                self.buffer[1:],
                stereo_feat[None, :, :],
            ],
            axis=0,
        )

    def get_frames(self):
        self.update()

        x = torch.from_numpy(
            self.buffer.copy()
        ).unsqueeze(0)

        return x.float()

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None