"""Stage 3b: 局所パッチ学習 (BCE multi-label + KD + EWC)。

設計書 Phase 3 セクション 4 に準拠。

実行例:
    python train_stage3b.py --config configs/stage3b.yaml \\
        --stage3a_ckpt outputs/stage3a/stage3a_last.pt
"""

import argparse
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Phase3_Training ディレクトリを sys.path に追加 (datasets/ models/ losses/ を解決するため)
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))  # Phase3_Training/ 自身を追加

# PatchAlign3D 参照リポジトリ
import types as _types
_REF = _HERE.parents[1] / "_reference_repos" / "PatchAlign3D" / "src"
if str(_REF) not in sys.path:
    sys.path.insert(0, str(_REF))

# patchalign3d を仮想パッケージとして登録 (Phase1 と同じアプローチ)
if "patchalign3d" not in sys.modules:
    _pa3d = _types.ModuleType("patchalign3d")
    _pa3d.__path__ = [str(_REF)]
    sys.modules["patchalign3d"] = _pa3d

import yaml
from easydict import EasyDict

from phase3_datasets.dental_dataset import DentalPatchDataset, collate_dental
from phase3_models.stage3_model import build_stage3_models, freeze_encoder_except_last_block
from phase3_models.text_cache import build_clip_and_cache
from phase3_models.ewc import EWC, compute_fisher_from_loader
from phase3_losses.bce_multilabel import compute_patch_label_distribution, dental_bce_loss
from phase3_losses.distillation import kd_patch_mse, kd_text_kl


# ---------------------------------------------------------------------------
# ユーティリティ
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# EWC 初期化 (Stage 3a チェックポイントから)
# ---------------------------------------------------------------------------

def build_ewc_from_stage3a(
    student,
    proj,
    text_cache,
    train_loader,
    device: torch.device,
    ewc_fisher_path: str,
    n_fisher_batches: int = 50,
    temp_val: float = 14.3,
) -> EWC | None:
    """Stage 3a 完了後の Fisher を計算して EWC を構築。

    ewc_fisher_path が存在する場合はキャッシュを使用する。
    """
    p = Path(ewc_fisher_path)
    if p.exists():
        logging.getLogger("stage3b").info(f"EWC Fisher をキャッシュからロード: {p}")
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        return EWC.from_dict(d)

    logging.getLogger("stage3b").info(
        f"Fisher 情報行列を計算中 ({n_fisher_batches} バッチ)..."
    )
    # Stage 3a パラメータを保存
    params_3a = {
        n: param.clone().detach()
        for n, param in student.named_parameters()
        if param.requires_grad
    }

    fisher = compute_fisher_from_loader(
        student, proj, text_cache, train_loader, device,
        n_batches=n_fisher_batches, temp_val=temp_val,
    )

    ewc = EWC.from_params_and_fisher(params_3a, fisher)
    torch.save(ewc.to_dict(), str(p))
    logging.getLogger("stage3b").info(f"EWC Fisher を保存: {p}")
    return ewc


# ---------------------------------------------------------------------------
# 1 エポック学習
# ---------------------------------------------------------------------------

def train_one_epoch(
    student,
    teacher,
    proj,
    temp_module,
    text_cache,
    ewc: EWC | None,
    loader,
    optimizer,
    device,
    cfg: EasyDict,
    epoch: int,
) -> dict:
    student.train()
    proj.train()
    teacher.eval()

    lambda1 = cfg.loss.get("lambda_bce",    1.0)
    lambda2 = cfg.loss.get("lambda_kd",     0.3)
    lambda3 = cfg.loss.get("lambda_ewc",    1000.0)
    lambda4 = cfg.loss.get("lambda_kd_text", 0.0)  # KL テキスト蒸留 (オプション)

    total_loss = total_bce = total_kd = total_ewc = 0.0
    n_batches = 0
    total_correct = total_patches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        if not batch:
            continue

        points = torch.cat([prepare_points(s, device) for s in batch], dim=0)  # (B, 3, N)

        optimizer.zero_grad()

        # Student forward
        patch_emb_s, _, patch_idx = student.forward_patches(points)   # (B, 384, G), (B, G, M)
        patch_feat_s = proj(patch_emb_s)                               # (B, G, D)

        # Teacher forward (no_grad)
        with torch.no_grad():
            patch_emb_t, _, _ = teacher.forward_patches(points)
            patch_feat_t = proj(patch_emb_t)

        scale = temp_module()
        batch_loss_bce  = torch.tensor(0.0, device=device)
        batch_loss_kd_t = torch.tensor(0.0, device=device)
        valid_count = 0

        for b_idx, sample in enumerate(batch):
            names_b = sample.get("label_names", [])
            if not names_b:
                continue

            text_feats_b = text_cache.encode_raw_texts(names_b).to(device)  # (K, D)
            if text_feats_b.numel() == 0:
                continue

            lm = sample["label_masks"].to(device)                     # (K, N)
            loss_bce_b, Y_patch = dental_bce_loss(
                patch_feat_s[b_idx], text_feats_b, lm, patch_idx[b_idx], scale
            )
            batch_loss_bce = batch_loss_bce + loss_bce_b
            valid_count += 1

            # KL テキスト蒸留 (オプション)
            if lambda4 > 0.0:
                with torch.no_grad():
                    logits_t = patch_feat_t[b_idx] @ text_feats_b.T * scale
                logits_s = patch_feat_s[b_idx] @ text_feats_b.T * scale
                kl = kd_text_kl(logits_s, logits_t, temperature=4.0)
                batch_loss_kd_t = batch_loss_kd_t + kl

            # パッチ精度計算
            with torch.no_grad():
                logits_eval = patch_feat_s[b_idx].detach() @ text_feats_b.T * scale
                pred = logits_eval.argmax(dim=-1)
                point_lbl = sample.get("point_labels", None)
                if point_lbl is not None:
                    from phase3_losses.infonce import _majority_label_per_patch
                    hard = _majority_label_per_patch(
                        point_lbl.to(device), patch_idx[b_idx], len(names_b)
                    )
                    valid_p = hard >= 0
                    if valid_p.any():
                        total_correct += (pred[valid_p] == hard[valid_p]).sum().item()
                        total_patches += valid_p.sum().item()

        if valid_count == 0:
            continue

        # KD (パッチ特徴 MSE)
        kd_loss = kd_patch_mse(patch_feat_s, patch_feat_t)

        # EWC ペナルティ
        ewc_loss = ewc.penalty(student) if ewc is not None else torch.tensor(0.0, device=device)

        total = (
            lambda1 * (batch_loss_bce / valid_count)
            + lambda2 * kd_loss
            + lambda3 * ewc_loss
            + lambda4 * (batch_loss_kd_t / valid_count)
        )

        total.backward()
        optimizer.step()

        total_loss += total.item()
        total_bce  += (batch_loss_bce.item() / valid_count)
        total_kd   += kd_loss.item()
        total_ewc  += ewc_loss.item() if isinstance(ewc_loss, torch.Tensor) else 0.0
        n_batches  += 1

    avg = lambda x: x / max(n_batches, 1)
    patch_acc = total_correct / max(total_patches, 1)
    return {
        "loss":      avg(total_loss),
        "bce":       avg(total_bce),
        "kd":        avg(total_kd),
        "ewc":       avg(total_ewc),
        "patch_acc": patch_acc,
        "temp":      temp_module().item(),
    }


# ---------------------------------------------------------------------------
# 評価
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_one_epoch(
    student,
    proj,
    temp_module,
    text_cache,
    loader,
    device,
    epoch: int,
) -> dict:
    student.eval()
    proj.eval()

    total_loss = 0.0
    total_correct = total_patches = 0
    n_batches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Val]", leave=False):
        if not batch:
            continue

        points = torch.cat([prepare_points(s, device) for s in batch], dim=0)
        patch_emb, _, patch_idx = student.forward_patches(points)
        patch_feat = proj(patch_emb)
        scale = temp_module()

        for b_idx, sample in enumerate(batch):
            names_b = sample.get("label_names", [])
            if not names_b:
                continue
            text_feats_b = text_cache.encode_raw_texts(names_b).to(device)
            if text_feats_b.numel() == 0:
                continue

            lm = sample["label_masks"].to(device)
            loss_b, _ = dental_bce_loss(
                patch_feat[b_idx], text_feats_b, lm, patch_idx[b_idx], scale
            )
            total_loss += loss_b.item()
            n_batches += 1

            # パッチ精度
            logits = patch_feat[b_idx] @ text_feats_b.T * scale
            pred = logits.argmax(dim=-1)
            point_lbl = sample.get("point_labels", None)
            if point_lbl is not None:
                from phase3_losses.infonce import _majority_label_per_patch
                hard = _majority_label_per_patch(
                    point_lbl.to(device), patch_idx[b_idx], len(names_b)
                )
                valid_p = hard >= 0
                if valid_p.any():
                    total_correct += (pred[valid_p] == hard[valid_p]).sum().item()
                    total_patches += valid_p.sum().item()

    return {
        "val_loss":  total_loss / max(n_batches, 1),
        "patch_acc": total_correct / max(total_patches, 1),
    }


# ---------------------------------------------------------------------------
# メイン
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser("Stage 3b: 局所パッチ BCE + KD + EWC 学習")
    parser.add_argument("--config",      type=str, default="configs/stage3b.yaml")
    parser.add_argument("--stage3a_ckpt", type=str, default="",
                        help="Stage 3a の最終チェックポイント (stage3a_last.pt)")
    parser.add_argument("--gpu",         type=str, default=None)
    parser.add_argument("--resume",      type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    # ---- ログ設定 ----
    out_dir = Path(cfg.get("output_dir", "outputs/stage3b"))
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
    logger = logging.getLogger("stage3b")
    logger.info(f"Config: {cfg}")
    logger.info(f"Device: {device}")

    # ---- データセット ----
    ds_cfg = cfg.data
    train_ds = DentalPatchDataset(
        ply_dir       = ds_cfg.ply_dir,
        json_path     = ds_cfg.json_path,
        npoints       = ds_cfg.get("npoints", 2048),
        mode          = "3b",
        split         = "train",
        val_ratio     = ds_cfg.get("val_ratio", 0.2),
        seed          = cfg.get("seed", 42),
        text_augment  = ds_cfg.get("text_augment", True),
        point_augment = ds_cfg.get("point_augment", False),
    )
    val_ds = DentalPatchDataset(
        ply_dir       = ds_cfg.ply_dir,
        json_path     = ds_cfg.json_path,
        npoints       = ds_cfg.get("npoints", 2048),
        mode          = "3b",
        split         = "val",
        val_ratio     = ds_cfg.get("val_ratio", 0.2),
        seed          = cfg.get("seed", 42),
        text_augment  = False,      # 検証は固定テキスト
        point_augment = False,      # 検証時は拡張なし
    )
    logger.info(f"Train: {len(train_ds)} 患者  Val: {len(val_ds)} 患者")

    tr_cfg = cfg.training
    train_loader = DataLoader(
        train_ds,
        batch_size  = tr_cfg.get("batch_size", 4),
        shuffle     = True,
        num_workers = tr_cfg.get("num_workers", 0),
        collate_fn  = collate_dental,
        pin_memory  = device.type == "cuda",
    )
    val_loader = DataLoader(
        val_ds,
        batch_size  = tr_cfg.get("batch_size", 4),
        shuffle     = False,
        num_workers = 0,
        collate_fn  = collate_dental,
    )

    # ---- CLIP & テキストキャッシュ ----
    clip_cfg = cfg.clip
    _, text_cache = build_clip_and_cache(
        clip_model_name = clip_cfg.get("model", "ViT-bigG-14"),
        clip_pretrained = clip_cfg.get("pretrained", "laion2b_s39b_b160k"),
        device          = device,
        capacity        = clip_cfg.get("cache_capacity", 10000),
    )

    # ---- モデル ----
    stage3a_ckpt = args.stage3a_ckpt or cfg.model.get("stage3a_ckpt", "")
    student, teacher, proj, temp_module = build_stage3_models(
        cfg       = cfg.model,
        device    = device,
        ckpt_path = stage3a_ckpt,
    )
    # Stage 3a チェックポイントから重みを上書き
    if stage3a_ckpt and Path(stage3a_ckpt).exists():
        ckpt_3a = torch.load(stage3a_ckpt, map_location="cpu", weights_only=False)
        student.load_state_dict(ckpt_3a["student"], strict=False)
        proj.load_state_dict(ckpt_3a["proj"], strict=False)
        if "temp" in ckpt_3a:
            temp_module.load_state_dict(ckpt_3a["temp"])
        logger.info(f"Stage 3a チェックポイントをロード: {stage3a_ckpt}")
        # Teacher を Stage 3a 完了重みで更新
        teacher.load_state_dict(student.state_dict(), strict=True)
        teacher.eval()

    # ---- EWC 構築 ----
    ewc: EWC | None = None
    if cfg.loss.get("lambda_ewc", 0.0) > 0:
        ewc_path = str(out_dir / "ewc_fisher.pt")
        ewc = build_ewc_from_stage3a(
            student, proj, text_cache,
            train_loader, device, ewc_path,
            n_fisher_batches = cfg.loss.get("ewc_fisher_batches", 50),
            temp_val         = temp_module().item(),
        )
        logger.info("EWC を構築しました。")

    # ---- オプティマイザ ----
    params = (
        list(filter(lambda p: p.requires_grad, student.parameters()))
        + list(proj.parameters())
        + list(temp_module.parameters())
    )
    optimizer = torch.optim.AdamW(
        params,
        lr           = tr_cfg.get("lr", 3e-5),
        weight_decay = tr_cfg.get("weight_decay", 0.05),
    )

    n_epochs  = tr_cfg.get("epochs", 50)
    n_warmup  = tr_cfg.get("warmup_epochs", 5)

    def lr_lambda(epoch_0idx: int) -> float:
        ep = epoch_0idx + 1
        if ep <= n_warmup:
            return ep / n_warmup
        progress = (ep - n_warmup) / max(n_epochs - n_warmup, 1)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---- 再開 ----
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

    # ---- 学習ループ ----
    eval_every = tr_cfg.get("eval_every", 5)
    save_every = tr_cfg.get("save_every", 10)

    for epoch in range(start_epoch, n_epochs + 1):
        train_metrics = train_one_epoch(
            student, teacher, proj, temp_module, text_cache, ewc,
            train_loader, optimizer, device, cfg, epoch
        )
        scheduler.step()

        logger.info(
            f"Epoch {epoch:3d}/{n_epochs}"
            f"  loss={train_metrics['loss']:.4f}"
            f"  bce={train_metrics['bce']:.4f}"
            f"  kd={train_metrics['kd']:.4f}"
            f"  ewc={train_metrics['ewc']:.4e}"
            f"  patch_acc={train_metrics['patch_acc']:.3f}"
            f"  temp={train_metrics['temp']:.2f}"
            f"  lr={scheduler.get_last_lr()[0]:.2e}"
        )

        # 検証
        if epoch % eval_every == 0 or epoch == n_epochs:
            val_metrics = eval_one_epoch(
                student, proj, temp_module, text_cache,
                val_loader, device, epoch
            )
            val_loss = val_metrics["val_loss"]
            logger.info(
                f"  -> val_loss={val_loss:.4f}"
                f"  val_patch_acc={val_metrics['patch_acc']:.3f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            state = {
                "epoch":         epoch,
                "student":       student.state_dict(),
                "proj":          proj.state_dict(),
                "temp":          temp_module.state_dict(),
                "optimizer":     optimizer.state_dict(),
                "scheduler":     scheduler.state_dict(),
                "best_val_loss": best_val_loss,
                "train_metrics": train_metrics,
                "val_metrics":   val_metrics,
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

    # ---- 最終チェックポイント ----
    final_path = out_dir / "stage3b_best.pt"
    best_ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location="cpu", weights_only=False)
    torch.save(
        {
            "student": best_ckpt["student"],
            "proj":    best_ckpt["proj"],
            "epoch":   best_ckpt["epoch"],
        },
        final_path,
    )
    logger.info(f"Stage 3b 完了。最良チェックポイント: {final_path}")


if __name__ == "__main__":
    main()
