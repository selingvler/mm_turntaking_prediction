#!/usr/bin/env python3
"""
Rigorous streaming-vs-offline comparison.

Builds a STREAMING vap timeline (each frame predicted from the most recent
window whose right edge sits `setback` seconds ahead of that frame — i.e. the
causal live read with `setback` lookahead latency), then scores it with the
SAME s_pred_p_now / shift_hold_p_now AUC as selinc/validate.py, restricted to
handovers toward the silenced robot channel (--incoming-channel 1).

This answers: does the live signal reproduce the offline AUC (~0.94), and if so
at what lookahead latency? One model pass populates every setback at once.

    python live_service/eval_streaming.py                       # pretrained
    python live_service/eval_streaming.py --weights <ft>.pt
"""
import argparse, json, pickle, sys, tempfile
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "selinc"))
import run_early_fusion as ref                                   # noqa: E402
from predictor import ShiftPredictor, AUDIO_SR, LABEL_FPS        # noqa: E402
from dataset_management.dataset_manager.scripts.get_tt_events import get_events_from_tg  # noqa: E402
from turn_taking.analysis.validation.validation import get_preds  # noqa: E402
from sklearn.metrics import roc_auc_score                         # noqa: E402
import validate as V                                              # noqa: E402

SETBACKS_S = [0.0, 0.5, 1.0, 2.0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio",    default="processed_data/aeeee708.wav")
    ap.add_argument("--video",    default="raw_data/mediapipe_output/synced_video_60feat_stereo_bs.pkl")
    ap.add_argument("--textgrid", default="processed_data/aeeee708.TextGrid")
    ap.add_argument("--weights",  default=None)
    ap.add_argument("--hop",      type=float, default=0.25)
    ap.add_argument("--window",   type=float, default=20.0)
    ap.add_argument("--incoming-channel", type=int, default=1)
    a = ap.parse_args()
    A = lambda p: p if Path(p).is_absolute() else ROOT / p

    pred = ShiftPredictor(weights_path=a.weights, window_s=a.window, fill_std=0.01)
    video_arr, vfps = ref.load_video(A(a.video))
    dur_s = video_arr.shape[0] / vfps
    audio = ref.load_audio_aligned(A(a.audio), dur_s, sr_target=AUDIO_SR)

    win_a = int(round(a.window * AUDIO_SR)); win_v = int(round(a.window * vfps))
    hop_a = int(round(a.hop * AUDIO_SR));    hop_v = int(round(a.hop * vfps))
    hop_l = int(round(a.hop * LABEL_FPS))
    T_lab = int(round(dur_s * LABEL_FPS))
    Wl    = int(round(a.window * LABEL_FPS))                      # 1000

    # streaming vap timelines, one per setback
    vap_tl = {sb: np.zeros((T_lab, 256), np.float32) for sb in SETBACKS_S}
    vad_tl = np.zeros((T_lab, 2), np.float32)
    n_steps = max(0, (video_arr.shape[0] - win_v) // hop_v + 1)
    print(f"streaming {n_steps} hops @ {a.hop}s, window {a.window}s, "
          f"populating setbacks {SETBACKS_S}")

    with torch.no_grad():
        for i in range(n_steps):
            va, vv = i * hop_a, i * hop_v
            a_win = audio[va:va + win_a].copy(); v_win = video_arr[vv:vv + win_v].copy()
            if a_win.shape[0] < win_a or v_win.shape[0] < win_v:
                break
            a_win[:, 1] = pred._fill_robot_audio(a_win.shape[0]); v_win[:, :, 1] = 0.0
            af = torch.from_numpy(np.ascontiguousarray(a_win)).float().unsqueeze(0)
            ff = torch.from_numpy(np.ascontiguousarray(v_win)).float().unsqueeze(0)
            vad, vap = pred.model({"audio_chunk": af, "frames": ff})
            vap = torch.softmax(vap, dim=-1).squeeze(0).cpu().numpy()   # (Wl,256)
            vad = torch.sigmoid(vad).squeeze(0).cpu().numpy()           # (Wl,2)
            sf  = int(round(vv / vfps * LABEL_FPS))                     # global start frame
            for sb in SETBACKS_S:
                nf  = int(round(sb * LABEL_FPS))
                hi  = Wl - 1 - nf                                       # newest emittable local idx
                lo  = max(0, hi - hop_l + 1)                            # tile by one hop
                for lp in range(lo, hi + 1):
                    g = sf + lp
                    if 0 <= g < T_lab:
                        vap_tl[sb][g] = vap[lp]
                        if sb == SETBACKS_S[0]:
                            vad_tl[g] = vad[lp]
            if (i + 1) % 200 == 0 or i + 1 == n_steps:
                print(f"  {i+1}/{n_steps}", flush=True)

    # score each setback with validate's exact s_pred / shift_hold AUC
    print(f"\n=== streaming s_pred_p_now / shift_hold_p_now AUC "
          f"(incoming=ch{a.incoming_channel}) ===")
    print(f"{'setback':>8} {'s_pred_now':>11} {'shift_hold_now':>15} {'s_pred_fut':>11}")
    for sb in SETBACKS_S:
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            pickle.dump([torch.from_numpy(vap_tl[sb]).float(),
                         torch.from_numpy(vad_tl).float()],
                        open(tmp / "aeeee708.pkl", "wb"))
            events = get_events_from_tg(str(A(a.textgrid)))
            json.dump(events, open(tmp / "aeeee708.json", "w"))
            V.filter_events_by_incoming(tmp / "aeeee708.json", a.incoming_channel)
            yt, ys = get_preds(["aeeee708"], str(tmp), str(tmp))
        def auc(task):
            y = yt["ekstedt_events"].get(task, []); s = ys["ekstedt_events"].get(task, [])
            return roc_auc_score(y, s) if len(set(y)) == 2 else float("nan")
        # operating threshold on the STREAMING score distribution (not EP0 0.58)
        y = np.array(yt["ekstedt_events"].get("s_pred_p_now", []))
        s = np.array(ys["ekstedt_events"].get("s_pred_p_now", []))
        thr_str = "n/a"
        if len(set(y.tolist())) == 2:
            from sklearn.metrics import precision_recall_curve, f1_score
            p, r, th = precision_recall_curve(y, s)
            f1s = 2*p*r/np.clip(p+r, 1e-9, None)
            bi = int(np.nanargmax(f1s[:-1])) if len(th) else 0
            t_star = float(th[bi]); f1m = float(f1s[bi])
            thr_str = f"thr*={t_star:.3f} F1={f1m:.2f}"
        print(f"{sb:>6.1f}s  {auc('s_pred_p_now'):>11.3f}  "
              f"{auc('shift_hold_p_now'):>15.3f}  {auc('s_pred_p_future'):>11.3f}   {thr_str}")
    print("\n(offline merged-timeline reference: s_pred_p_now ~0.94; EP0 thr 0.58 does NOT "
          "transfer — use thr* above, calibrated on the streaming score scale)")


if __name__ == "__main__":
    main()
