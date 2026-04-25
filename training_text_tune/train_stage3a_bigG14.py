"""Stage 3a: 領域パッチ学習 (InfoNCE + KD) — ViT-bigG-14 版。

train_stage3a.py と同一ロジック。デフォルト設定を ViT-bigG-14 用に変更。
テキストエンコーダは凍結したまま点群エンコーダのみを学習する。

実行例:
    python train_stage3a_bigG14.py --config configs/stage3a_bigG14.yaml
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import types as _types
_REF = _HERE.parents[1] / "_reference_repos" / "PatchAlign3D" / "src"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

if "patchalign3d" not in sys.modules:
    _pa3d = _types.ModuleType("patchalign3d")
    _pa3d.__path__ = [str(_REF)]
    sys.modules["patchalign3d"] = _pa3d

import yaml
from easydict import EasyDict

from phase3_datasets.dental_dataset import DentalPatchDataset, collate_dental
from phase3_models.stage3_model import build_stage3_models, freeze_encoder_except_last_block
from phase3_models.text_cache import build_clip_and_cache
from phase3_losses.infonce import infonce_loss_batch
from phase3_losses.distillation import kd_patch_mse


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> EasyDict:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return EasyDict(raw)


def prepare_points(sample: dict, device: torch.device) -> torch.Tensor:
    return sample["points"].unsqueeze(0).to(device, non_blocking=True)


def train_one_epoch(
    student,
    teacher,
    proj,
    temp_module,
    text_cache,
    loader,
    optimizer,
    device,
    cfg: EasyDict,
    epoch: int,
    logger: logging.Logger,
) -> dict:
    student.train()
    proj.train()
    teacher.eval()

    lambda1 = cfg.loss.get("lambda_infonce", 1.0)
    lambda2 = cfg.loss.get("lambda_kd",      0.5)

    total_loss = total_infonce = total_kd = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        if not batch:
            continue

        points = torch.cat([prepare_points(s, device) for s in batch], dim=0)

        optimizer.zero_grad()

        patch_emb_s, _, patch_idx = student.forward_patches(points)
        patch_feat_s = proj(patch_emb_s)

        with torch.no_grad():
            patch_emb_t, _, _ = teacher.forward_patches(points)
            patch_feat_t = proj(patch_emb_t)

        all_names: list[str] = []
        for s in batch:
            all_names.extend(s["label_names"])

        if not all_names:
            continue

        unique_names = list(dict.fromkeys(all_names))
        text_feats_all = text_cache.encode_raw_texts(unique_names).to(device)
        name_to_idx = {n: i for i, n in enumerate(unique_names)}

        scale = temp_module()
        batch_loss = torch.tensor(0.0, device=device)
        valid_count = 0

        for b_idx, s in enumerate(batch):
            names_b = s["label_names"]
            if not names_b:
                continue
            idx_b = [name_to_idx[n] for n in names_b]
            text_feats_b = text_feats_all[idx_b]

            point_lbl = s["point_labels"].to(device)

            loss_infonce_b = infonce_loss_batch(
                patch_feat_s[b_idx:b_idx+1],
                text_feats_b,
                [point_lbl],
                patch_idx[b_idx:b_idx+1],
                len(names_b),
                scale,
            )
            batch_loss = batch_loss + lambda1 * loss_infonce_b
            total_infonce += loss_infonce_b.item()
            valid_count += 1

        kd_loss = kd_patch_mse(patch_feat_s, patch_feat_t)
        batch_loss = batch_loss + lambda2 * kd_loss * max(valid_count, 1)
        total_kd += kd_loss.item()

        if valid_count > 0:
            (batch_loss / valid_count).backward()
            optimizer.step()
            total_loss += (batch_loss.item() / valid_count)
            n_batches += 1

    avg = lambda x: x / max(n_batches, 1)
    return {
        "loss":    avg(total_loss),
        "infonce": avg(total_infonce),
        "kd":      avg(total_kd),
        "temp":    temp_module().item(),
    }


@torch.no_grad()
def eval_one_epoch(
    student,
    proj,
    temp_module,
    text_cache,
    loader,
    device,
    cfg: EasyDict,
    epoch: int,
) -> dict:
    student.eval()
    proj.eval()

    total_loss = 0.0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        if not batch:
            continue

        points = torch.cat([prepare_points(s, device) for s in batch], dim=0)
        patch_emb, _, patch_idx = student.forward_patches(points)
        patch_feat = proj(patch_emb)

        scale = temp_module()
        for b_idx, s in enumerate(batch):
            names_b = s["label_names"]
            if not names_b:
                continue
            text_feats_b = text_cache.encode_raw_texts(names_b).to(device)
            point_lbl = s["point_labels"].to(device)

            loss_b = infonce_loss_batch(
                patch_feat[b_idx:b_idx+1],
                text_feats_b,
                [point_lbl],
                patch_idx[b_idx:b_idx+1],
                len(names_b),
                scale,
            )
            total_loss += loss_b.item()
            n_batches += 1

    return {"val_loss": total_loss / max(n_batches, 1)}


def main():
    parser = argparse.ArgumentParser("Stage 3a: 領域パッチ InfoNCE 学習 (ViT-bigG-14)")
    parser.add_argument("--config", type=str, default="configs/stage3a_bigG14.yaml")
    parser.add_argument("--gpu",    type=str, default=None)
    parser.add_argument("--resume", type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg.get("output_dir", "outputs/stage3a_bigG14"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(str(out_dir / "train.log")),
            logging.StreamHandler(),
        ],
    )
    logger = logging.getLogger("stage3a_bigG14")
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    ds_cfg = cfg.data
    train_ds = DentalPatchDataset(
        ply_dir       = ds_cfg.ply_dir,
        json_path     = ds_cfg.json_path,
        npoints       = ds_cfg.get("npoints", 2048),
        mode          = "3a",
        split         = "train",
        val_ratio     = ds_cfg.get("val_ratio", 0.2),
        seed          = cfg.get("seed", 42),
        text_augment  = False,
        point_augment = ds_cfg.get("point_augment", False),
    )
    val_ds = DentalPatchDataset(
        ply_dir       = ds_cfg.ply_dir,
        json_path     = ds_cfg.json_path,
        npoints       = ds_cfg.get("npoints", 2048),
        mode          = "3a",
        split         = "val",
        val_ratio     = ds_cfg.get("val_ratio", 0.2),
        seed          = cfg.get("seed", 42),
        text_augment  = False,
        point_augment = False,
    )
    logger.info(f"Train: {len(train_ds)} 患者  Val: {len(val_ds)} 患者")

    train_loader = DataLoader(
        train_ds,
        batch_size  = cfg.training.get("batch_size", 4),
        shuffle     = True,
        num_workers = cfg.training.get("num_workers", 0),
        collate_fn  = collate_dental,
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = cfg.training.get("batch_size", 4),
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_dental,
    )

    clip_cfg = cfg.clip
    _, text_cache = build_clip_and_cache(
        clip_model_name = clip_cfg.get("model", "ViT-bigG-14"),
        clip_pretrained = clip_cfg.get("pretrained", "laion2b_s39b_b160k"),
        device          = device,
        capacity        = clip_cfg.get("cache_capacity", 5000),
    )

    student, teacher, proj, temp_module = build_stage3_models(
        cfg       = cfg.model,
        device    = device,
        ckpt_path = cfg.model.get("stage2_ckpt", ""),
    )

    tr_cfg = cfg.training
    params = (
        list(filter(lambda p: p.requires_grad, student.parameters()))
        + list(proj.parameters())
        + list(temp_module.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr           = tr_cfg.get("lr", 1e-4),
        weight_decay = tr_cfg.get("weight_decay", 0.05),
    )
    n_epochs = tr_cfg.get("epochs", 30)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=tr_cfg.get("lr_min", 1e-6)
    )

    start_epoch = 1
    best_val_loss = float("inf")
    if args.resume:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt["student"])
        proj.load_state_dict(ckpt["proj"])
        temp_module.load_state_dict(ckpt["temp"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"チェックポイントを再開: epoch={start_epoch-1}")

    eval_every = tr_cfg.get("eval_every", 5)
    save_every = tr_cfg.get("save_every", 10)

    for epoch in range(start_epoch, n_epochs + 1):
        train_metrics = train_one_epoch(
            student, teacher, proj, temp_module, text_cache,
            train_loader, optimizer, device, cfg, epoch, logger
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs}"
            f"  loss={train_metrics['loss']:.4f}"
            f"  infonce={train_metrics['infonce']:.4f}"
            f"  kd={train_metrics['kd']:.4f}"
            f"  temp={train_metrics['temp']:.2f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        if epoch % eval_every == 0 or epoch == n_epochs:
            val_metrics = eval_one_epoch(
                student, proj, temp_module, text_cache,
                val_loader, device, cfg, epoch
            )
            val_loss = val_metrics["val_loss"]
            logger.info(f"  -> val_loss={val_loss:.4f}")

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            state = {
                "epoch":          epoch,
                "student":        student.state_dict(),
                "proj":           proj.state_dict(),
                "temp":           temp_module.state_dict(),
                "optimizer":      optimizer.state_dict(),
                "scheduler":      scheduler.state_dict(),
                "best_val_loss":  best_val_loss,
                "train_metrics":  train_metrics,
                "val_metrics":    val_metrics,
            }
            torch.save(state, ckpt_dir / "last.pt")
            if is_best:
                torch.save(state, ckpt_dir / "best.pt")
                logger.info(f"  -> Best チェックポイントを保存 (val_loss={val_loss:.4f})")

        if epoch % save_every == 0:
            torch.save(
                {"epoch": epoch, "student": student.state_dict(), "proj": proj.state_dict()},
                ckpt_dir / f"epoch_{epoch:03d}.pt",
            )

    final_path = out_dir / "stage3a_last.pt"
    torch.save(
        {
            "epoch":   n_epochs,
            "student": student.state_dict(),
            "proj":    proj.state_dict(),
            "temp":    temp_module.state_dict(),
        },
        final_path,
    )
    logger.info(f"Stage 3a 完了。最終チェックポイント: {final_path}")
    logger.info("次: train_stage3b_bigG14.py --init_ckpt <このパス> を実行してください。")


if __name__ == "__main__":
    main()
