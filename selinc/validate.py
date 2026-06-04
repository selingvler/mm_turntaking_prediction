#!/usr/bin/env python3
"""
Event-level validation for the early-fusion (audio + video) turn-taking model.

Reads run_early_fusion.py's prediction dict directly, overlap-averages the
per-window VAP outputs onto a single 50 Hz timeline, then runs the same
threshold-optimised F1 / Balanced-Acc tables as seling_scripts/validate_seling.py
— with semantic class labels for each task, sample counts, and optimal
thresholds shown alongside.
"""

import argparse
import copy
import json
import math
import os
import pickle
import sys
import tempfile
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn import metrics


# CUDA disable (Mac CPU)
if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False


# torchaudio 2.11 dropped sox_io_backend; the turn_taking import chain still
# references it at module level. Stub so the import completes; never actually
# invoked for pkl-based validation.
_fake_backend = types.ModuleType("torchaudio.backend")
_fake_sox     = types.ModuleType("torchaudio.backend.sox_io_backend")
def _fake_info(*a, **kw):
    raise RuntimeError("sox_io_backend stub — should never be called")
_fake_sox.info = _fake_info
sys.modules.setdefault("torchaudio.backend", _fake_backend)
sys.modules.setdefault("torchaudio.backend.sox_io_backend", _fake_sox)


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from turn_taking.analysis.validation.validation import (   # noqa: E402
    get_preds,
    optimal_thresholds,
    apply_thresholds,
    apply_score,
    f1_score_func,
    f1_score_func_detailed,
)
from dataset_management.dataset_manager.scripts.get_tt_events import (   # noqa: E402
    get_events_from_tg,
)


# Semantic class labels per sub_event (matches validate_seling.py).
TASK_LABELS = {
    "shift_hold_p_future": ("hold", "shift"),
    "shift_hold_p_now":    ("hold", "shift"),

    "s_pred_p_future":     ("negative", "shift"),
    "s_pred_p_now":        ("negative", "shift"),

    "backchannel":         ("non-bc", "backchannel"),

    "short_long":          ("long", "short"),

    "overlaps_before_p_now":     ("overlap_hold", "overlap_shift"),
    "overlaps_before_p_future":  ("overlap_hold", "overlap_shift"),

    "overlaps_after_p_now":      ("overlap_hold", "overlap_shift"),
    "overlaps_after_p_future":   ("overlap_hold", "overlap_shift"),

    "overlap_spred_before_p_future": ("negative", "overlap_shift"),

    "gap_0_p_future":      ("hold", "shift"),
    "gap_0_p_now":         ("hold", "shift"),

    "superset_p_future":   ("hold_like", "shift_like"),
}


def ensure_groundtruth_json(json_path: Path, textgrid_path: Path,
                            audio_offset_s: float = 0.0):
    if json_path.exists():
        return
    if not textgrid_path.exists():
        sys.exit(f"missing both JSON and TextGrid: {json_path} / {textgrid_path}")

    # If the prediction timeline starts at original_t = audio_offset_s (because
    # run_early_fusion.py skipped the first audio_offset_s of the .wav), shift
    # the TextGrid intervals by -audio_offset_s before extracting events so GT
    # and prediction share the same t=0. The shifted .TextGrid is written into
    # json_path's directory — caller is expected to invoke us from within a
    # tempfile.TemporaryDirectory so nothing persists on disk.
    if audio_offset_s > 0:
        import textgrid as tg_lib
        tg = tg_lib.TextGrid.fromFile(str(textgrid_path))
        for tier in tg.tiers:
            new_intervals = []
            for iv in tier.intervals:
                new_min = float(iv.minTime) - audio_offset_s
                new_max = float(iv.maxTime) - audio_offset_s
                if new_max <= 0:
                    continue  # interval entirely inside the skipped intro
                iv.minTime = max(0.0, new_min)
                iv.maxTime = new_max
                new_intervals.append(iv)
            tier.intervals = new_intervals
            tier.minTime = 0.0
            tier.maxTime = max(float(tier.maxTime) - audio_offset_s, 0.0)
        tg.minTime = 0.0
        tg.maxTime = max(float(tg.maxTime) - audio_offset_s, 0.0)
        shifted = json_path.parent / f"{textgrid_path.stem}_shifted.TextGrid"
        tg.write(str(shifted))
        textgrid_path = shifted
        print(f"  shifted TextGrid by {audio_offset_s}s (in tempdir, auto-deleted)")

    print(f"generating ground-truth JSON from {textgrid_path.name} ...")
    events = get_events_from_tg(str(textgrid_path))
    with open(json_path, "w") as f:
        json.dump(events, f, indent=4)


def merge_vap_windows(vap_win, starts_s, win_s, fps, total_T):
    """Overlap-average per-window softmax VAP outputs (Nw, win_frames, 256)
    onto a single (total_T, 256) timeline at fps Hz."""
    feat = vap_win.shape[-1]
    acc = np.zeros((total_T, feat), dtype=np.float64)
    cnt = np.zeros((total_T, 1),    dtype=np.float64)
    win_frames = int(round(win_s * fps))
    for c, t0 in zip(vap_win, starts_s):
        s = int(round(float(t0) * fps))
        e = min(s + win_frames, total_T)
        n = e - s
        if n <= 0:
            continue
        acc[s:e] += c[:n]
        cnt[s:e] += 1.0
    cnt = np.clip(cnt, 1.0, None)
    return (acc / cnt).astype(np.float32)


def load_ef_predictions(ef_pkl: Path, win_s: float = 20.0):
    """Read run_early_fusion.py's dict pkl, return
    (vap (T,256), vad (T,2), audio_offset_s) where audio_offset_s is the
    AUDIO_OFFSET_S the inference script used (== how far the prediction
    timeline's t=0 sits inside the original recording)."""
    with open(ef_pkl, "rb") as f:
        ef = pickle.load(f)
    fps     = int(ef["label_fps"])
    starts  = ef["window_starts_s_in_video"]
    vap_win = ef["vap_windows"]
    vad_arr = ef["vad_timeline_prob"]
    audio_offset_s = float(ef.get("audio_offset_s", 0.0))
    T = vad_arr.shape[0]
    vap_arr = merge_vap_windows(vap_win, starts, win_s, fps, T)
    return (torch.from_numpy(vap_arr).float(),
            torch.from_numpy(vad_arr).float(),
            audio_offset_s)


def evaluate_final_detailed(file_id, ef_pkl, output_dir, textgrid_path):
    print(f"\nEvaluating file: {file_id}")

    try:
        if not ef_pkl.exists():
            print(f"early-fusion pkl missing: {ef_pkl}")
            return

        output_dir.mkdir(parents=True, exist_ok=True)
        vap_t, vad_t, audio_offset_s = load_ef_predictions(ef_pkl)
        print(f"  vap={tuple(vap_t.shape)}  vad={tuple(vad_t.shape)}  "
              f"audio_offset={audio_offset_s}s")

        # get_preds() reads {id}.pkl + {id}.json from a directory, so stage
        # the converted prediction + GT json in a tempdir.
        with tempfile.TemporaryDirectory() as tmp_str:
            tmp = Path(tmp_str)
            with open(tmp / f"{file_id}.pkl", "wb") as f:
                pickle.dump([vap_t, vad_t], f)
            ensure_groundtruth_json(tmp / f"{file_id}.json", textgrid_path,
                                    audio_offset_s)

            y_true, y_score = get_preds([file_id], str(tmp), str(tmp))

        if not y_true:
            print("No matching prediction/json pair found.")
            return

        print("\nOptimizing thresholds for weighted F1...")
        y_score_f1 = copy.deepcopy(y_score)
        thresholds_f1 = optimal_thresholds(y_true, y_score_f1,
                                           func_to_optimise=f1_score_func)
        y_pred_f1 = apply_thresholds(y_score_f1, thresholds_f1)
        f1_results = apply_score(y_true, y_pred_f1, f1_score_func_detailed)

        y_score_bal = copy.deepcopy(y_score)
        thresholds_bal = optimal_thresholds(y_true, y_score_bal,
                                            func_to_optimise=metrics.balanced_accuracy_score)
        y_pred_bal = apply_thresholds(y_score_bal, thresholds_bal)
        bal_acc_results = apply_score(y_true, y_pred_bal, metrics.balanced_accuracy_score)

        out_lines = []
        header = "=" * 70
        title  = f"MM-VAP VALIDATION RESULTS :: {file_id}"
        print("\n" + header)
        print(title)
        print(header)
        out_lines += [header, title, header]

        for event_class in y_true.keys():
            ec_header = f"\n[{event_class.upper()}]"
            print(ec_header)
            out_lines.append(ec_header)

            for sub_event in y_true[event_class].keys():
                if len(y_true[event_class][sub_event]) == 0:
                    continue

                weighted_f1, class0_f1, class1_f1 = f1_results[event_class][sub_event][0]
                bal_acc        = bal_acc_results[event_class][sub_event][0]
                threshold_used = thresholds_f1[event_class][sub_event]
                n_samples      = len(y_true[event_class][sub_event])

                label0, label1 = TASK_LABELS.get(sub_event, ("class0", "class1"))

                block = (
                    f"\n  Task: {sub_event}\n"
                    f"    Samples        : {n_samples}\n"
                    f"    Threshold      : {threshold_used:.4f}\n"
                    f"    Weighted F1    : {weighted_f1:.4f}\n"
                    f"    Balanced Acc   : {bal_acc:.4f}\n"
                    f"    F1 [{label0}]  : {class0_f1:.4f}\n"
                    f"    F1 [{label1}]  : {class1_f1:.4f}"
                )
                print(block)
                out_lines.append(block)

        out_path = output_dir / f"{file_id}_validation.txt"
        out_path.write_text("\n".join(out_lines) + "\n")
        print(f"\nsaved: {out_path}")

    except Exception as e:
        print(f"\nERROR: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate early-fusion MM-VAP predictions.")
    parser.add_argument("file_id", nargs="?", default="aeeee708_trimmed",
                        help="File ID without extension")
    parser.add_argument("--ef-pkl", default="selinc/predictions_early_fusion.pkl",
                        help="Path to run_early_fusion.py's prediction dict pkl")
    parser.add_argument("--output-dir", default="my_outputs",
                        help="Where to write {file_id}_validation.txt")
    parser.add_argument("--textgrid", default=None,
                        help="TextGrid used to build GT events. "
                             "Default: processed_data/{file_id}.TextGrid")
    args = parser.parse_args()

    ef_pkl = Path(args.ef_pkl)
    if not ef_pkl.is_absolute():
        ef_pkl = ROOT / ef_pkl
    output_dir = ROOT / args.output_dir
    textgrid = Path(args.textgrid) if args.textgrid else (
        ROOT / "processed_data" / f"{args.file_id}.TextGrid"
    )

    evaluate_final_detailed(args.file_id, ef_pkl, output_dir, textgrid)
