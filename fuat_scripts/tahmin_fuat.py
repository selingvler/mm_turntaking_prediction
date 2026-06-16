import os
import sys
import torch
import yaml
import time
import argparse
import numpy as np
from pathlib import Path

import pickle
from torch.utils.data import Dataset
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# --- CUDA YAMASI ---
if not torch.cuda.is_available():
    torch.cuda.set_device = lambda device: None
    torch.cuda.is_available = lambda: False
# -------------------

from dataset_management.dataset_manager.src.audio_manager import AudioManager
from turn_taking.analysis.validation.run_model import run_model_unbatched
from turn_taking.analysis.validation.probabilities import VAPDecoder
from turn_taking.model.model import StereoTransformerModel

try:
    import sounddevice as sd
except ImportError:
    sd = None

# Config dosyasının içindeki gizli "cuda" ayarlarını Mac'e uyarlayan yardımcı fonksiyon
def fix_config_device(d, target_device):
    if isinstance(d, dict):
        for k, v in d.items():
            if v == "cuda" or v == "cuda:0":
                d[k] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)
    elif isinstance(d, list):
        for i, v in enumerate(d):
            if v == "cuda" or v == "cuda:0":
                d[i] = str(target_device)
            elif isinstance(v, (dict, list)):
                fix_config_device(v, target_device)


class PredictionAudioDataset(Dataset):
    def __init__(self, audio_file: str, sr: int, feature_sr: int, window_size: int, step_size: int):
        super().__init__()

        self.audio_file = audio_file
        self.sr = sr
        self.feature_sr = feature_sr
        self.window_size = window_size
        self.step_size = step_size
        self.audio_normalize = True

        self.id = os.path.basename(audio_file).split('.')[0]
        self.load_audio()

    def load_audio(self):
        audio, _ = AudioManager.load_waveform(self.audio_file, self.sr, normalize=self.audio_normalize)
        size = int(self.sr * self.window_size)
        step = int(self.sr * self.step_size)

        audio = audio.unfold(0, size, step)
        self.audio = audio.permute(0, 2, 1)

    def __getitem__(self, index):
        return {
            "id": [self.id],
            "audio_chunk": self.audio[index, :],
        }

    def __len__(self):
        return self.audio.shape[0] - 1


def select_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Mac M1 (MPS) hizlandiricisi kullaniliyor.")
    else:
        device = torch.device("cpu")
        print("Uyari: MPS bulunamadi, CPU kullanilacak.")
    return device


def load_model(model_weights_path: str, cfg_path: str, device: torch.device):
    print("Model mimarisi kuruluyor...")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Config dosyasi bulunamadi: {cfg_path}")

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    # Config icindeki olasi 'cuda' parametrelerini temizle
    fix_config_device(cfg, device)

    model = StereoTransformerModel(cfg=cfg)

    print("Egitilmis agirliklar yukleniyor...")
    checkpoint = torch.load(model_weights_path, map_location=device)

    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    return model


def _safe_value(x):
    if hasattr(x, "item"):
        return float(x.item())
    return float(x)
_last_human_vad = 0.0
_silence_count = 0

def live_decision(vad_frame, p_future_frame, p_bc_frame, vad_threshold=0.5):
    global _last_human_vad, _silence_count

    vad_left = _safe_value(vad_frame[0])     # human
    vad_right = _safe_value(vad_frame[1])    # robot

    p_human = _safe_value(p_future_frame[0])
    p_robot = _safe_value(p_future_frame[1])

    bc_robot = _safe_value(p_bc_frame[1])

    was_speaking = _last_human_vad > 0.35
    now_silent = vad_left < 0.50

    if now_silent:
        _silence_count += 1
    else:
        _silence_count = 0

    _last_human_vad = vad_left

    # Robot zaten konuşuyorsa bekle
    if vad_right > 0.35:
        return "hold", vad_right

    # İnsan konuşmayı yeni bitirdiyse robot sırayı alabilir
    if was_speaking and _silence_count >= 1:
        return "shift", 0.65

    # İnsan konuşuyorsa robot beklesin
    if vad_left >= vad_threshold:

        if bc_robot > 0.015:
            return "backchannel", bc_robot

        if p_robot > 0.30 and vad_left < 0.85:
            return "shift", max(p_robot, 0.55)

        return "hold", p_human

    # Başlangıç sessizliğinde robot hemen atlamasın
    if _silence_count < 2:
        return "hold", 1.0 - vad_left

    # Uzun sessizlikte, robot tarafı makul görünüyorsa konuşabilir
    if p_robot > 0.30:
        return "shift", max(p_robot, 0.55)

    return "hold", 1.0 - vad_left


def stream_live_predictions(
    model,
    audio_path: str,
    start_time: float,
    end_time: float,
    sr: int,
    window_size: float,
    step_size: float,
    feature_hz: int,
    realtime: bool,
    print_only_events: bool,
):
    audio, _ = AudioManager.load_waveform(audio_path, sr, normalize=True)

    total_samples = audio.shape[0]
    start_sample = max(0, int(start_time * sr))
    end_sample = min(int(end_time * sr), total_samples)

    if end_sample <= start_sample:
        raise ValueError("end_time, start_time'dan buyuk olmalidir.")

    audio = audio[start_sample:end_sample, :]
    segment_samples = audio.shape[0]

    chunk_samples = int(window_size * sr)
    step_samples = int(step_size * sr)

    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("window_size ve step_size 0'dan buyuk olmalidir.")
    if segment_samples < chunk_samples:
        raise ValueError("Secilen aralik, window_size'dan kucuk olamaz.")

    decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
    device = next(model.parameters()).device

    chunk_idx = 0
    print("Canli tahmin basladi...")
    print("t=seconds | decision | conf | p_shift | p_hold | p_backchannel | vad_l | vad_r")

    printed_idx = 0
    for start in range(0, segment_samples - chunk_samples + 1, step_samples):
        stop = start + chunk_samples
        chunk = audio[start:stop, :]

        batch = {"audio_chunk": chunk.unsqueeze(0).to(device)}

        with torch.no_grad():
            vad_logits, vap_logits = model(batch)
            vad = torch.sigmoid(vad_logits).cpu()  # [1, T, 2]
            vap = torch.softmax(vap_logits, dim=-1).cpu()  # [1, T, C]

        p_future = decoder.p_future(vap).squeeze(0)
        p_bc = decoder.p_bc(vap).squeeze(0)

        vad_frame = vad[0, -1, :]
        p_future_frame = p_future[-1, :]
        p_bc_frame = p_bc[-1, :]

        decision, conf = live_decision(vad_frame, p_future_frame, p_bc_frame)

        current_t = start_time + (start + chunk_samples) / sr

        shift_score = _safe_value(p_future_frame[1 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 0])
        hold_score = _safe_value(p_future_frame[0 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 1])
        bc_score = _safe_value(p_bc_frame[1 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 0])

        if (not print_only_events) or decision in ("shift", "backchannel"):
            print(
                f"t={current_t:6.2f}s | {decision:11s} | {conf:5.3f} | "
                f"{shift_score:7.3f} | {hold_score:6.3f} | {bc_score:13.3f} | "
                f"{_safe_value(vad_frame[0]):5.3f} | {_safe_value(vad_frame[1]):5.3f}"
            )
            printed_idx += 1

        if realtime:
            # Pace output to simulate realtime streaming from file.
            time.sleep(step_size)

        chunk_idx += 1

    print(f"Canli tahmin tamamlandi. Islenen pencere sayisi: {chunk_idx}, Yazdirilan olay sayisi: {printed_idx}")


def stream_live_microphone(
    model,
    start_time: float,
    end_time: float,
    sr: int,
    window_size: float,
    step_size: float,
    print_only_events: bool,
):
    if sd is None:
        raise RuntimeError(
            "sounddevice paketi bulunamadi. Kurulum icin: pip install sounddevice"
        )

    capture_duration = end_time - start_time
    if capture_duration <= 0:
        raise ValueError("Mikrofon modu icin end_time, start_time'dan buyuk olmalidir.")

    chunk_samples = int(window_size * sr)
    step_samples = int(step_size * sr)
    if chunk_samples <= 0 or step_samples <= 0:
        raise ValueError("window_size ve step_size 0'dan buyuk olmalidir.")
    if capture_duration < window_size:
        raise ValueError("Secilen aralik, window_size'dan kucuk olamaz.")

    decoder = VAPDecoder(bin_times=[0.2, 0.4, 0.6, 0.8])
    device = next(model.parameters()).device

    # Ring buffer for latest mono mic samples.
    buffer = np.zeros((0,), dtype=np.float32)

    print("Mikrofon canli tahmin basladi...")
    print("t=seconds | decision | conf | p_shift | p_hold | p_backchannel | vad_l | vad_r")

    printed_idx = 0
    chunk_idx = 0
    start_wall = time.time()

    with sd.InputStream(
        samplerate=sr,
        channels=1,
        dtype="float32",
        blocksize=step_samples,
    ) as stream:
        while True:
            elapsed = time.time() - start_wall
            if elapsed >= capture_duration:
                break

            data, overflowed = stream.read(step_samples)
            if overflowed:
                pass

            mono = data[:, 0]
            buffer = np.concatenate((buffer, mono), axis=0)
            if buffer.shape[0] > chunk_samples:
                buffer = buffer[-chunk_samples:]
            if buffer.shape[0] < chunk_samples:
                continue

            # Build stereo: channel-0 = mic, channel-1 = zeros.
            stereo = np.stack((buffer, np.zeros_like(buffer)), axis=1)
            chunk = torch.from_numpy(stereo).float()

            batch = {"audio_chunk": chunk.unsqueeze(0).to(device)}
            with torch.no_grad():
                vad_logits, vap_logits = model(batch)
                vad = torch.sigmoid(vad_logits).cpu()
                vap = torch.softmax(vap_logits, dim=-1).cpu()

            p_future = decoder.p_future(vap).squeeze(0)
            p_bc = decoder.p_bc(vap).squeeze(0)

            vad_frame = vad[0, -1, :]
            p_future_frame = p_future[-1, :]
            p_bc_frame = p_bc[-1, :]

            decision, conf = live_decision(vad_frame, p_future_frame, p_bc_frame)
            current_t = elapsed

            shift_score = _safe_value(
                p_future_frame[1 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 0]
            )
            hold_score = _safe_value(
                p_future_frame[0 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 1]
            )
            bc_score = _safe_value(
                p_bc_frame[1 if _safe_value(vad_frame[0]) >= _safe_value(vad_frame[1]) else 0]
            )

            if (not print_only_events) or decision in ("shift", "backchannel"):
                print(
                    f"t={current_t:6.2f}s | {decision:11s} | {conf:5.3f} | "
                    f"{shift_score:7.3f} | {hold_score:6.3f} | {bc_score:13.3f} | "
                    f"{_safe_value(vad_frame[0]):5.3f} | {_safe_value(vad_frame[1]):5.3f}"
                )
                printed_idx += 1

            chunk_idx += 1

    print(
        f"Mikrofon canli tahmin tamamlandi. Islenen pencere sayisi: {chunk_idx}, "
        f"Yazdirilan olay sayisi: {printed_idx}"
    )

def run_batch_mode(args):
    # 1. Dosya Yollari
    output_predictions_dir = args.output_dir
    os.makedirs(output_predictions_dir, exist_ok=True)

    device = select_device()
    model = load_model(args.model_weights, args.config_path, device)

    print("Ses dosyasi isleniyor ve tahminler uretiliyor...")
    with torch.no_grad():
        dataset = PredictionAudioDataset(
            audio_file=args.audio_path,
            sr=args.sr,
            feature_sr=args.feature_hz,
            window_size=args.window_size,
            step_size=args.step_size,
        )
        vaps, vads = run_model_unbatched(
            model=model,
            dataset=dataset,
            mask_vad=False,
            feature_extraction_hz=args.feature_hz,
            window_size=args.window_size,
            step_size=args.step_size,
            mode='VAP'
        )

        file_id = os.path.basename(args.audio_path).split('.')[0]
        output_file = os.path.join(output_predictions_dir, f"{file_id}.pkl")
        with open(output_file, "wb") as f:
            pickle.dump([vaps, vads], f)

    print(f"Islem basariyla tamamlandi. Sonuclar '{output_predictions_dir}' klasorune kaydedildi.")


def run_live_mode(args):
    device = select_device()
    model = load_model(args.model_weights, args.config_path, device)

    if args.input_source == "mic":
        stream_live_microphone(
            model=model,
            start_time=args.start_time,
            end_time=args.end_time,
            sr=args.sr,
            window_size=args.window_size,
            step_size=args.step_size,
            print_only_events=args.print_only_events,
        )
    else:
        stream_live_predictions(
            model=model,
            audio_path=args.audio_path,
            start_time=args.start_time,
            end_time=args.end_time,
            sr=args.sr,
            window_size=args.window_size,
            step_size=args.step_size,
            feature_hz=args.feature_hz,
            realtime=args.realtime,
            print_only_events=args.print_only_events,
        )


def parse_args():
    parser = argparse.ArgumentParser(description="VAP tahmin scripti (batch + live)")
    parser.add_argument("--mode", choices=["batch", "live"], default="batch")
    parser.add_argument("--input_source", choices=["wav", "mic"], default="wav")
    parser.add_argument("--audio_path", default="processed_data/bau_veri_toplama_04.wav")
    parser.add_argument("--output_dir", default="processed_data/")
    parser.add_argument("--start_time", type=float, default=0.0, help="Live mode baslangic zamani (saniye)")
    parser.add_argument("--end_time", type=float, default=30.0, help="Live mode bitis zamani (saniye)")
    parser.add_argument("--window_size", type=float, default=4.0)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--sr", type=int, default=16_000)
    parser.add_argument("--feature_hz", type=int, default=50)
    parser.add_argument(
        "--model_weights",
        default="acl_sample_data_models/sample_trained_models/VAP_candor/20240822_105427_fold_0_epoch_10",
    )
    parser.add_argument(
        "--config_path",
        default="acl_sample_data_models/sample_trained_models/VAP_candor/20240822_105427_params.yaml",
    )
    parser.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Live modda adimlar arasinda bekleyerek gercek zamanli akisi simule et",
    )
    parser.add_argument(
        "--print_only_events",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sadece shift/backchannel kararlarini yazdir",
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    if args.mode == "live":
        run_live_mode(args)
    else:
        run_batch_mode(args)