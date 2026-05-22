import os
import yaml
import torch
import einops
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset_management.dataset_manager.dataloader.dataloader import AudioVisualDataset
from turn_taking.model.multimodal_model import EarlyVAFusion


DEVICE = "cpu"

CFG_PATH = "acl_sample_data_models/sample_trained_models/early_fusion_candor/config.yaml"
CKPT_PATH = "fine_tune_trained_models/merged_audio_ft_video_base.pt"

FOLD_DIR = "buse_workspace/folds/fold_session05"
WAV_DIR = "buse_workspace/all_wavs"
VIDEO_DIR = "buse_workspace/video_training_format"
CHANNELMAPS = "buse_workspace/video_training_format/channelmaps.pkl"

OUT_DIR = "fine_tune_trained_models/video_fusion_finetune_fold_session05"
os.makedirs(OUT_DIR, exist_ok=True)


def move_tensor(x):
    if torch.is_tensor(x):
        return x.to(DEVICE)
    return x


def make_loss(vad_true, vad_pred, vap_true, vap_pred):
    vap_pred = einops.rearrange(vap_pred, "b n d -> (b n) d")
    vap_true = einops.rearrange(vap_true, "b n -> (b n)")
    vap_loss = F.cross_entropy(vap_pred, vap_true)

    vad_loss = F.binary_cross_entropy_with_logits(vad_pred, vad_true)

    return vap_loss + vad_loss, vap_loss, vad_loss


def freeze_audio_encoder(model):
    frozen = 0
    trainable = 0

    for name, param in model.named_parameters():
        if name.startswith("audio_encoder"):
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    print(f"Frozen audio params: {frozen:,}")
    print(f"Trainable params    : {trainable:,}")


def resize_frames_to_model_video_len(frames):
    target_len = 600

    if frames.shape[1] == target_len:
        return frames

    b, t, d, c = frames.shape
    x = frames.permute(0, 2, 3, 1).reshape(b, d * c, t)

    x = torch.nn.functional.interpolate(
        x,
        size=target_len,
        mode="linear",
        align_corners=False,
    )

    x = x.reshape(b, d, c, target_len).permute(0, 3, 1, 2)
    return x


def make_batch(batch):
    id_, start, end, vad, vap, inverse_vap, audio_chunk, ch_l, ch_r, frames = batch

    audio_chunk = move_tensor(audio_chunk)
    frames = move_tensor(frames)
    frames = resize_frames_to_model_video_len(frames)

    return {
        "id": id_,
        "start_time": move_tensor(start),
        "end_time": move_tensor(end),
        "vad": move_tensor(vad),
        "vap": move_tensor(vap),
        "inverse_vap": move_tensor(inverse_vap),
        "audio_chunk": audio_chunk,
        "channel_ids": [ch_l, ch_r],
        "frames": frames,
    }


def run_epoch(model, loader, optimizer=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_vap = 0.0
    total_vad = 0.0

    for step, batch in enumerate(loader, start=1):
        batch = make_batch(batch)

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(is_train):
            vad_pred, vap_pred = model(batch)
            loss, vap_loss, vad_loss = make_loss(
                batch["vad"], vad_pred,
                batch["vap"], vap_pred
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()

        total_loss += loss.item()
        total_vap += vap_loss.item()
        total_vad += vad_loss.item()

        if step % 10 == 0:
            print(f"step {step}/{len(loader)} loss={loss.item():.4f}")

    n = max(1, len(loader))
    return total_loss / n, total_vap / n, total_vad / n


def main():
    print("Loading config...")
    cfg = yaml.safe_load(open(CFG_PATH))
    cfg["device"] = DEVICE
    cfg["training_cfg"]["batch_size"] = 1
    cfg["training_cfg"]["learning_rate"] = 1e-5
    cfg["training_cfg"]["n_epochs"] = 1
    cfg["training_cfg"]["flip_channel"] = 0.0
    cfg["training_cfg"]["mask_vad"] = 0

    print("Building model...")
    model = EarlyVAFusion(cfg=cfg).to(DEVICE)

    print("Loading merged checkpoint...")
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE)
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]

    missing, unexpected = model.load_state_dict(ckpt, strict=False)
    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    freeze_audio_encoder(model)

    train_ds = AudioVisualDataset(
        video_directory=VIDEO_DIR,
        video_format=".pkl",
        channelmaps=CHANNELMAPS,
        pickle_file=f"{FOLD_DIR}/train",
        mode="VAP",
        wavdir=WAV_DIR,
    )

    val_ds = AudioVisualDataset(
        video_directory=VIDEO_DIR,
        video_format=".pkl",
        channelmaps=CHANNELMAPS,
        pickle_file=f"{FOLD_DIR}/val",
        mode="VAP",
        wavdir=WAV_DIR,
    )

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-5,
        weight_decay=0.001,
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches  : {len(val_loader)}")

    for epoch in range(1, cfg["training_cfg"]["n_epochs"] + 1):
        print(f"\n===== EPOCH {epoch} =====")

        train_loss, train_vap, train_vad = run_epoch(model, train_loader, optimizer)
        print(f"TRAIN loss={train_loss:.4f} vap={train_vap:.4f} vad={train_vad:.4f}")

        val_loss, val_vap, val_vad = run_epoch(model, val_loader, optimizer=None)
        print(f"VAL   loss={val_loss:.4f} vap={val_vap:.4f} vad={val_vad:.4f}")

        out_path = os.path.join(OUT_DIR, f"audio_frozen_video_fusion_epoch_{epoch}.pt")
        torch.save(model.state_dict(), out_path)
        print("Saved:", out_path)


if __name__ == "__main__":
    main()
