#!/usr/bin/env python3
"""
Run the trained early-fusion (audio + video) turn-taking model on:
  audio:      processed_data/aeeee708.wav        (stereo, 16 kHz, ~26m40s)
  textgrid:   processed_data/aeeee708.TextGrid   (per-speaker word intervals)
  video pkl:  raw_data/mediapipe_output/<chosen>.pkl
              (T, 60, 2) stereo features built by build_60feat_pkl.py

Alignment
---------
The video pkl already starts 60 s into the recording (mediapipe_feature_extraction
skipped the first minute) and covers VIDEO_DUR_S seconds of conversation.
Here we trim BOTH the audio AND the textgrid VAD to exactly that same window,
so all three streams (audio / video / labels) are sample-locked:
    t = 0  in pkl  ==  t = 60 s in original recording
                   ==  start of the trimmed audio
                   ==  start of the trimmed VAD ground truth

Sliding-window inference
------------------------
We chunk the aligned audio + video into 20 s windows:
  audio chunk : 20 s × 16 000 Hz = 320 000 samples,  shape (20s_samples, 2)
  video chunk : 20 s × 16 fps   = 320 frames,        shape (320, 60, 2)
The model internally upsamples video 16 fps → 50 fps and pairs it with
50 Hz audio embeddings.

Validation
----------
After inference we
  1. merge the per-window VAD predictions onto a single 50 Hz timeline,
     averaging overlapped frames;
  2. build the ground-truth VAD at 50 Hz from the trimmed TextGrid;
  3. compute frame-level F1 / precision / recall / accuracy / ROC-AUC for
     each speaker and overall, plus a confusion matrix.

The script is CPU-only (Mac). It monkey-patches torch's `.to('cuda')`
calls used inside multimodal_model.py to redirect to CPU.

Outputs:
  selinc/predictions_early_fusion.pkl  - all of the above + raw outputs
  selinc/metrics_early_fusion.txt      - human-readable metrics summary
"""

# ------------------------------------------------------------------
# CUDA → CPU monkey patches (run BEFORE importing turn_taking modules)
# ------------------------------------------------------------------
import torch

def _pick_device():
    if torch.cuda.is_available():         return torch.device("cuda")
    if torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")

_TARGET_DEVICE = _pick_device()


def _redir_device(arg):
    """Map cuda-flavoured device arguments to _TARGET_DEVICE."""
    if arg is None:
        return arg
    if isinstance(arg, str):
        return str(_TARGET_DEVICE) if arg.startswith("cuda") else arg
    if isinstance(arg, torch.device):
        return _TARGET_DEVICE if arg.type == "cuda" else arg
    return arg


_orig_module_to = torch.nn.Module.to


def _patched_module_to(self, *args, **kwargs):
    args = tuple(_redir_device(a) for a in args)
    if "device" in kwargs:
        kwargs["device"] = _redir_device(kwargs["device"])
    return _orig_module_to(self, *args, **kwargs)


_orig_tensor_to = torch.Tensor.to


def _patched_tensor_to(self, *args, **kwargs):
    args = tuple(_redir_device(a) for a in args)
    if "device" in kwargs:
        kwargs["device"] = _redir_device(kwargs["device"])
    return _orig_tensor_to(self, *args, **kwargs)


torch.nn.Module.to = _patched_module_to
torch.Tensor.to = _patched_tensor_to

# Disable any cuda probing that might fail or stall.
if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False

# ------------------------------------------------------------------

import os
import pickle
import sys
import yaml
from pathlib import Path

import numpy as np
import soundfile as sf
import textgrid as tg_lib

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turn_taking.model.multimodal_model import EarlyVAFusion  # noqa: E402


# ------------------------------------------------------------------
# Hardcoded paths — adjust here, no CLI args.
#
# All three input streams (audio / textgrid / video pkl) have already had
# their first 60 seconds removed:
#   * processed_data/aeeee708_trimmed.wav        — ffmpeg -ss 60 ...
#   * processed_data/aeeee708_trimmed.TextGrid   — interval times shifted -60 s
#   * raw_data/mediapipe_output/synced_video_60feat_stereo_bs.pkl
#         (mediapipe_feature_extraction.py skipped the first minute already)
# So all three start at t = 0.
# ------------------------------------------------------------------
# All four input paths and the OUT_TAG suffix are env-var overridable so we
# can swap recordings (aeeee708 vs session02) and feature variants (bs vs
# non-bs) without editing this file.
TAG             = os.environ.get("OUT_TAG", "")
AUDIO_PATH      = Path(os.environ.get("AUDIO_PATH") or
                       ROOT / "processed_data" / "aeeee708.wav")
TEXTGRID_PATH   = Path(os.environ.get("TEXTGRID_PATH") or
                       ROOT / "processed_data" / "aeeee708.TextGrid")
VIDEO_PKL_PATH  = Path(os.environ.get("VIDEO_PKL") or
                       ROOT / "raw_data" / "mediapipe_output" / "synced_video_60feat_stereo_bs.pkl")
CKPT_DIR        = ROOT / "acl_sample_data_models" / "sample_trained_models" / "early_fusion_candor"
CONFIG_PATH     = CKPT_DIR / "config.yaml"
# WEIGHTS_PATH env-var lets us swap in the Türkçe fine-tune merged weights
# (audio modules from fine-tuned StereoTransformerModel + video modules from
# original Candor early-fusion). Default = original Candor early-fusion.
WEIGHTS_PATH    = Path(os.environ.get("WEIGHTS_PATH") or (CKPT_DIR / "weights"))
OUT_DIR         = ROOT / "selinc"
OUT_PRED_PATH   = OUT_DIR / f"predictions_early_fusion{TAG}.pkl"
OUT_METRICS_PATH = OUT_DIR / f"metrics_early_fusion{TAG}.txt"

AUDIO_OFFSET_S  = float(os.environ.get("AUDIO_OFFSET_S", "0.0"))   # set to 60 for untrimmed wav+textgrid paired with mediapipe-skipped video
WINDOW_S        = 20.0   # model audio sequence_len (=1000) at 50 Hz
HOP_S           = 10.0   # 50% overlap
LABEL_FPS       = 50     # model output frame rate (audio embedding rate)
VAD_THRESHOLD   = 0.5    # threshold for converting probabilities → 0/1


def fix_config_device(d, target_device):
    """Replace any 'cuda*' string inside a (possibly nested) cfg dict/list."""
    if isinstance(d, dict):
        for k, v in d.items():
            if isinstance(v, str) and v.startswith("cuda"):
                d[k] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if isinstance(v, str) and v.startswith("cuda"):
                d[i] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)


def load_model() -> EarlyVAFusion:
    if not CONFIG_PATH.exists():
        sys.exit(f"config not found: {CONFIG_PATH}")
    if not WEIGHTS_PATH.exists():
        sys.exit(f"weights not found: {WEIGHTS_PATH}")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    fix_config_device(cfg, _TARGET_DEVICE)

    model = EarlyVAFusion(cfg=cfg)
    state = torch.load(str(WEIGHTS_PATH), map_location=_TARGET_DEVICE, weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    state = {(k[len("model."):] if k.startswith("model.") else k): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"missing keys ({len(missing)}): {missing[:5]}{' ...' if len(missing) > 5 else ''}")
    print(f"unexpected keys ({len(unexpected)}): {unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}")
    model.to(_TARGET_DEVICE).eval()
    return model


def load_audio_aligned(path: Path, video_dur_s: float, sr_target: int = 16_000) -> np.ndarray:
    """Read the wav, drop the first AUDIO_OFFSET_S seconds, then trim/pad to
    exactly video_dur_s seconds. Returns float32 array (samples, 2)."""
    info = sf.info(str(path))
    if info.samplerate != sr_target:
        sys.exit(f"audio sr {info.samplerate} != expected {sr_target}; "
                 f"resample beforehand or set sr_target")
    if info.channels != 2:
        sys.exit(f"audio must be stereo (got {info.channels} ch). Re-run "
                 f"merge_audio_seling.py to produce 2-channel wav.")

    start = int(round(AUDIO_OFFSET_S * sr_target))
    want  = int(round(video_dur_s    * sr_target))

    audio, _ = sf.read(str(path), start=start, frames=want, dtype="float32",
                       always_2d=True)
    if audio.shape[0] < want:
        # pad tail with zeros if recording is slightly shorter
        pad = np.zeros((want - audio.shape[0], audio.shape[1]), dtype=np.float32)
        audio = np.concatenate([audio, pad], axis=0)

    # Peak-normalize each channel independently, matching Candor training pipeline
    # (AudioManager.load_waveform normalize=True, threshold=0.05)
    for ch in range(audio.shape[1]):
        peak = np.abs(audio[:, ch]).max()
        if peak > 0.05:
            audio[:, ch] /= peak

    return audio


def load_video(path: Path, target_fps: float = 30.0) -> tuple[np.ndarray, float]:
    """Returns (arr (T_new, 60, 2), target_fps).

    The early-fusion checkpoint was trained on OpenFace features at 30 fps,
    and its internal upsampler hard-codes scale_factor = audio_seqlen /
    video_seqlen = 1000/600 = 1.667. So per 20 s window the model expects
    exactly 600 video frames (= 20 s @ 30 fps). MediaPipe gives us 16 fps,
    so we linearly interpolate every channel along the time axis from the
    source fps to target_fps once at load time.
    """
    with open(path, "rb") as f:
        d = pickle.load(f)
    arr = d["arr"].astype(np.float32)         # (T_src, 60, 2)
    src_fps = float(d["fps"])

    if abs(src_fps - target_fps) < 1e-3:
        return arr, src_fps

    import torch.nn.functional as F
    T_src = arr.shape[0]
    duration = T_src / src_fps
    T_new = int(round(duration * target_fps))

    # (T_src, 60, 2) -> (1, 120, T_src)  for F.interpolate (mode='linear')
    arr_t = torch.from_numpy(arr).float()
    arr_t = arr_t.permute(1, 2, 0).reshape(1, 60 * 2, T_src)
    arr_t = F.interpolate(arr_t, size=T_new, mode="linear", align_corners=True)
    arr_t = arr_t.reshape(60, 2, T_new).permute(2, 0, 1).contiguous()
    arr_new = arr_t.numpy()

    print(f"  resampled video: {arr.shape} @ {src_fps} fps -> "
          f"{arr_new.shape} @ {target_fps} fps")
    return arr_new, float(target_fps)


def windows(start_max: float, win_s: float, hop_s: float):
    """Yield window start times in seconds, last window ends at start_max."""
    s = 0.0
    while s + win_s <= start_max + 1e-6:
        yield s
        s += hop_s


def load_groundtruth_vad(textgrid_path: Path, video_dur_s: float,
                         audio_offset_s: float, label_fps: int) -> np.ndarray:
    """Parse the TextGrid, drop the first audio_offset_s seconds, and rasterize
    onto a label_fps grid for video_dur_s seconds. Returns (T, 2) float32 with
    1.0 where the speaker is talking, 0.0 otherwise.
    Tier order:  tier 0 -> speaker 0 (left / Selin)
                 tier 1 -> speaker 1 (right / Ahmet).
    """
    tg = tg_lib.TextGrid.fromFile(str(textgrid_path))
    if len(tg.tiers) < 2:
        sys.exit(f"TextGrid {textgrid_path} has <2 tiers; cannot build stereo VAD")

    T = int(round(video_dur_s * label_fps))
    out = np.zeros((T, 2), dtype=np.float32)

    for ch_idx in (0, 1):
        for interval in tg.tiers[ch_idx].intervals:
            if not interval.mark or interval.mark.strip() == "":
                continue
            start = float(interval.minTime) - audio_offset_s
            end   = float(interval.maxTime) - audio_offset_s
            if end <= 0 or start >= video_dur_s:
                continue
            start = max(0.0, start)
            end   = min(video_dur_s, end)
            s_idx = int(round(start * label_fps))
            e_idx = int(round(end   * label_fps))
            if e_idx > s_idx:
                out[s_idx:e_idx, ch_idx] = 1.0

    return out


def merge_window_vads(vad_chunks: list, window_starts_s: list,
                      win_s: float, label_fps: int,
                      total_T: int) -> np.ndarray:
    """Average overlapping windowed VAD predictions onto one timeline.
    Each chunk is (1000, 2) at label_fps Hz over win_s seconds.
    Returns (total_T, 2) float32."""
    acc = np.zeros((total_T, 2), dtype=np.float64)
    cnt = np.zeros((total_T, 2), dtype=np.float64)
    win_frames = int(round(win_s * label_fps))
    for vad_p, t0 in zip(vad_chunks, window_starts_s):
        s = int(round(t0 * label_fps))
        e = min(s + win_frames, total_T)
        n = e - s
        if n <= 0:
            continue
        acc[s:e] += vad_p[:n]
        cnt[s:e] += 1.0
    cnt = np.clip(cnt, 1.0, None)
    return (acc / cnt).astype(np.float32)


def compute_vad_metrics(gt: np.ndarray, pred_prob: np.ndarray,
                        threshold: float) -> dict:
    """Frame-level VAD metrics, per-speaker and pooled."""
    from sklearn.metrics import (
        f1_score, precision_score, recall_score, accuracy_score,
        roc_auc_score, confusion_matrix,
    )

    pred = (pred_prob >= threshold).astype(np.int8)
    gt_i = gt.astype(np.int8)

    out = {"threshold": threshold,
           "label_fps": LABEL_FPS,
           "n_frames": int(gt.shape[0])}

    for ch_idx, label in [(0, "left"), (1, "right")]:
        y_true = gt_i[:, ch_idx]
        y_pred = pred[:, ch_idx]
        y_prob = pred_prob[:, ch_idx]
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            auc = float("nan")  # only one class present
        out[label] = {
            "support_speaking_frames": int(y_true.sum()),
            "speaking_fraction": float(y_true.mean()),
            "f1":        float(f1_score(y_true, y_pred, zero_division=0)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall":    float(recall_score(y_true, y_pred, zero_division=0)),
            "accuracy":  float(accuracy_score(y_true, y_pred)),
            "roc_auc":   auc,
            "confusion_matrix": cm.tolist(),  # rows = true, cols = predicted
        }

    # pooled (flatten per-speaker into one big binary problem)
    yt = gt_i.reshape(-1)
    yp = pred.reshape(-1)
    yprob = pred_prob.reshape(-1)
    try:
        auc = float(roc_auc_score(yt, yprob))
    except ValueError:
        auc = float("nan")
    out["pooled"] = {
        "f1":        float(f1_score(yt, yp, zero_division=0)),
        "precision": float(precision_score(yt, yp, zero_division=0)),
        "recall":    float(recall_score(yt, yp, zero_division=0)),
        "accuracy":  float(accuracy_score(yt, yp)),
        "roc_auc":   auc,
    }
    return out


def format_metrics(m: dict) -> str:
    def fmt_block(label, d):
        return (
            f"  {label:7s}  "
            f"F1={d['f1']:.4f}  P={d['precision']:.4f}  R={d['recall']:.4f}  "
            f"Acc={d['accuracy']:.4f}  AUC={d['roc_auc']:.4f}"
        )

    lines = [
        f"Frame-level VAD validation",
        f"  threshold       : {m['threshold']}",
        f"  label fps       : {m['label_fps']}",
        f"  total frames    : {m['n_frames']}",
        "",
        f"  left speaking   : {m['left']['support_speaking_frames']} "
        f"frames ({100*m['left']['speaking_fraction']:.1f}%)",
        f"  right speaking  : {m['right']['support_speaking_frames']} "
        f"frames ({100*m['right']['speaking_fraction']:.1f}%)",
        "",
        fmt_block("left",   m["left"]),
        fmt_block("right",  m["right"]),
        fmt_block("pooled", m["pooled"]),
        "",
        "Confusion matrix (rows=true, cols=pred; [silent, speaking]):",
        f"  left : {m['left']['confusion_matrix']}",
        f"  right: {m['right']['confusion_matrix']}",
    ]
    return "\n".join(lines)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"loading video pkl: {VIDEO_PKL_PATH}")
    video_arr, video_fps = load_video(VIDEO_PKL_PATH)
    video_dur_s = video_arr.shape[0] / video_fps
    print(f"  video: shape={video_arr.shape}, fps={video_fps}, "
          f"duration={video_dur_s:.2f} s, starts at t={AUDIO_OFFSET_S}s")

    print(f"loading audio: {AUDIO_PATH}")
    audio = load_audio_aligned(AUDIO_PATH, video_dur_s, sr_target=16_000)
    print(f"  audio: shape={audio.shape}, duration="
          f"{audio.shape[0] / 16_000:.2f} s")

    # Sanity: durations must match within one frame.
    audio_dur_s = audio.shape[0] / 16_000.0
    if abs(audio_dur_s - video_dur_s) > 1.0 / video_fps:
        print(f"WARN: audio/video duration differ by "
              f"{abs(audio_dur_s - video_dur_s):.3f} s")

    print("loading model...")
    model = load_model()

    audio_per_win = int(round(WINDOW_S * 16_000))   # 320 000 samples
    video_per_win = int(round(WINDOW_S * video_fps))  # 320 frames @ 16 fps

    starts = list(windows(video_dur_s, WINDOW_S, HOP_S))
    print(f"running inference: {len(starts)} windows of {WINDOW_S}s, "
          f"hop {HOP_S}s")

    vad_chunks = []
    vap_chunks = []

    with torch.no_grad():
        for i, t0 in enumerate(starts):
            a_start = int(round(t0 * 16_000))
            v_start = int(round(t0 * video_fps))

            a_chunk = audio[a_start:a_start + audio_per_win]            # (S, 2)
            v_chunk = video_arr[v_start:v_start + video_per_win]        # (Nv, 60, 2)

            # pad if the tail window came up short (shouldn't happen with our
            # window generator but be safe)
            if a_chunk.shape[0] < audio_per_win:
                pad = np.zeros((audio_per_win - a_chunk.shape[0], 2),
                               dtype=np.float32)
                a_chunk = np.concatenate([a_chunk, pad], axis=0)
            if v_chunk.shape[0] < video_per_win:
                pad = np.zeros((video_per_win - v_chunk.shape[0], 60, 2),
                               dtype=np.float32)
                v_chunk = np.concatenate([v_chunk, pad], axis=0)

            audio_t = torch.from_numpy(a_chunk).float().unsqueeze(0)    # (1, S, 2)
            frames_t = torch.from_numpy(v_chunk).float().unsqueeze(0)   # (1, Nv, 60, 2)

            vad, vap = model({"audio_chunk": audio_t, "frames": frames_t})
            vad_p = torch.sigmoid(vad).squeeze(0).cpu().numpy()
            vap_p = torch.softmax(vap, dim=-1).squeeze(0).cpu().numpy()
            vad_chunks.append(vad_p)
            vap_chunks.append(vap_p)

            if (i + 1) % 5 == 0 or i + 1 == len(starts):
                print(f"  window {i+1}/{len(starts)}  t0={t0:.1f}s  "
                      f"vad mean L={vad_p[:, 0].mean():.3f} "
                      f"R={vad_p[:, 1].mean():.3f}", flush=True)

    # ----- merge window VAD onto a single 50 Hz timeline ------------
    total_T = int(round(video_dur_s * LABEL_FPS))
    pred_prob = merge_window_vads(vad_chunks, starts, WINDOW_S, LABEL_FPS, total_T)
    print(f"\nmerged VAD timeline: {pred_prob.shape}  ({total_T} frames @ {LABEL_FPS} Hz)")

    # ----- ground truth from TextGrid -------------------------------
    print(f"loading textgrid: {TEXTGRID_PATH}")
    gt = load_groundtruth_vad(TEXTGRID_PATH, video_dur_s,
                              AUDIO_OFFSET_S, LABEL_FPS)
    print(f"  ground-truth VAD: {gt.shape}, "
          f"L speaks {100*gt[:,0].mean():.1f}%  R speaks {100*gt[:,1].mean():.1f}%")

    # length safety: trim both to common length
    n = min(gt.shape[0], pred_prob.shape[0])
    gt        = gt[:n]
    pred_prob = pred_prob[:n]

    # ----- metrics --------------------------------------------------
    metrics = compute_vad_metrics(gt, pred_prob, threshold=VAD_THRESHOLD)
    metrics_text = format_metrics(metrics)
    print("\n" + metrics_text)

    with open(OUT_METRICS_PATH, "w") as f:
        f.write(metrics_text + "\n")
    print(f"\nsaved metrics: {OUT_METRICS_PATH}")

    out = {
        "vad_windows":        np.stack(vad_chunks, axis=0),    # raw per-window
        "vap_windows":        np.stack(vap_chunks, axis=0),
        "vad_timeline_prob":  pred_prob,                       # merged 50 Hz
        "vad_timeline_pred":  (pred_prob >= VAD_THRESHOLD).astype(np.int8),
        "vad_timeline_gt":    gt.astype(np.int8),
        "metrics":            metrics,
        "window_starts_s_in_video": np.array(starts, dtype=np.float64),
        "audio_offset_s":     AUDIO_OFFSET_S,
        "video_pkl":          str(VIDEO_PKL_PATH),
        "audio_path":         str(AUDIO_PATH),
        "textgrid_path":      str(TEXTGRID_PATH),
        "input_video_fps":    video_fps,
        "audio_sr":           16_000,
        "label_fps":          LABEL_FPS,
        "speakers":           ["left (0) Selin", "right (1) Ahmet"],
    }
    with open(OUT_PRED_PATH, "wb") as f:
        pickle.dump(out, f)
    print(f"saved predictions: {OUT_PRED_PATH}  "
          f"({OUT_PRED_PATH.stat().st_size/1e6:.1f} MB)")
    print(f"  vad_windows shape: {out['vad_windows'].shape}")
    print(f"  vap_windows shape: {out['vap_windows'].shape}")
    print(f"  vad_timeline shape: {out['vad_timeline_prob'].shape}")


if __name__ == "__main__":
    main()
