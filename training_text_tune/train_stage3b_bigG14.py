"""Stage 3b: 局所パッチ学習 (BCE + KD + EWC) — ViT-bigG-14 版。

train_stage3b.py をベースに ViT-bigG-14 テキストエンコーダに対応。
主な変更点:
  - テキストモジュールの train/eval 切り替えを ViT-bigG-14 構造に合わせる
    (clip_model.transformer + clip_model.ln_final)
  - text_encoder チェックポイントを {transformer, ln_final, text_projection} 形式で保存
  - オプティマイザのテキストパラメータ収集で visual encoder を除外

実行例:
    python train_stage3b_bigG14.py --config configs/stage3b_bigG14.yaml \\
        --init_ckpt ../../checkpoints/patchalign3d/patchalign3d.pt
"""

import argparse
import logging
import os
import random
import shutil
import sys
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
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
from phase3_models.stage3_model import (
    build_stage3_models,
    freeze_encoder_except_last_block,
    freeze_text_encoder_except_last,
)
from phase3_models.text_cache import build_clip_and_cache
from phase3_models.ewc import EWC, compute_fisher_from_loader
from phase3_losses.bce_multilabel import compute_patch_label_distribution, dental_bce_loss
from phase3_losses.distillation import kd_patch_mse, kd_text_kl


# ---------------------------------------------------------------------------
# ViT-bigG-14 テキストエンコーダの train/eval ヘルパー
# ---------------------------------------------------------------------------

def _set_vit_text_train_mode(clip_model, training: bool = True) -> None:
    """ViT-bigG-14 のテキスト関連モジュールの学習モードを切り替える。

    clip_model.transformer (CLIPTextTransformer) と clip_model.ln_final を対象にする。
    text_projection は Parameter のため train/eval の影響を受けない。
    """
    clip_model.transformer.train(training)
    if hasattr(clip_model, "ln_final"):
        clip_model.ln_final.train(training)


def _save_vit_text_encoder_state(clip_model) -> dict:
    """ViT-bigG-14 テキストエンコーダの状態を保存用辞書に変換する。"""
    state = {"transformer": clip_model.transformer.state_dict()}
    if hasattr(clip_model, "ln_final"):
        state["ln_final"] = clip_model.ln_final.state_dict()
    if hasattr(clip_model, "text_projection") and clip_model.text_projection is not None:
        state["text_projection"] = clip_model.text_projection.data.clone()
    return state


def _load_vit_text_encoder_state(clip_model, te: dict) -> None:
    """保存済み辞書から ViT-bigG-14 テキストエンコーダの状態を復元する。"""
    clip_model.transformer.load_state_dict(te["transformer"], strict=False)
    if hasattr(clip_model, "ln_final") and te.get("ln_final"):
        clip_model.ln_final.load_state_dict(te["ln_final"], strict=False)
    if (
        hasattr(clip_model, "text_projection")
        and clip_model.text_projection is not None
        and te.get("text_projection") is not None
    ):
        clip_model.text_projection.data.copy_(te["text_projection"])


# ---------------------------------------------------------------------------
# 非同期チェックポイント保存（Google Drive I/O を学習ループからオフロード）
# ---------------------------------------------------------------------------

_copy_thread: threading.Thread | None = None


def _save_with_local_buffer(
    state: dict,
    drive_path: Path,
    tmp_stem: str,
) -> threading.Thread:
    """/tmp に torch.save してから Drive へバックグラウンドコピーする。

    前回のコピースレッドが生きていれば join してから新スレッドを起動する。
    これにより Drive 書き込みが完了する前に次エポックが開始できる。
    """
    global _copy_thread
    if _copy_thread is not None and _copy_thread.is_alive():
        _copy_thread.join()

    tmp_path = Path(f"/tmp/{tmp_stem}.pt")
    torch.save(state, str(tmp_path))  # SSD への書き込みは ~0.5 秒

    t = threading.Thread(
        target=shutil.copy2,
        args=(str(tmp_path), str(drive_path)),
        daemon=True,
    )
    t.start()
    return t


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
# EWC 初期化
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
    p = Path(ewc_fisher_path)
    if p.exists():
        logging.getLogger("stage3b_bigG14").info(f"EWC Fisher をキャッシュからロード: {p}")
        d = torch.load(str(p), map_location="cpu", weights_only=False)
        return EWC.from_dict(d)

    logging.getLogger("stage3b_bigG14").info(
        f"Fisher 情報行列を計算中 ({n_fisher_batches} バッチ)..."
    )
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
    logging.getLogger("stage3b_bigG14").info(f"EWC Fisher を保存: {p}")
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
    clip_model=None,
    teacher_emb_cache: dict | None = None,
) -> dict:
    student.train()
    proj.train()
    teacher.eval()
    if clip_model is not None:
        _set_vit_text_train_mode(clip_model, training=True)

    lambda1 = cfg.loss.get("lambda_bce",    1.0)
    lambda2 = cfg.loss.get("lambda_kd",     0.3)
    lambda3 = cfg.loss.get("lambda_ewc",    1000.0)
    lambda4 = cfg.loss.get("lambda_kd_text", 0.0)

    total_loss = total_bce = total_kd = total_ewc = 0.0
    n_batches = 0
    total_correct = total_patches = 0

    for batch in tqdm(loader, desc=f"Epoch {epoch} [Train]", leave=False):
        if not batch:
            continue

        points = torch.cat([prepare_points(s, device) for s in batch], dim=0)

        optimizer.zero_grad()

        # バッチ内全テキストを一括収集 (8回 → 1回の CLIP forward)
        all_texts: list[str] = []
        text_slices: list[tuple[int, int]] = []
        for _s in batch:
            _names = _s.get("label_names", [])
            _si, _ei = len(all_texts), len(all_texts) + len(_names)
            text_slices.append((_si, _ei))
            all_texts.extend(_names)

        with torch.autocast("cuda", dtype=torch.bfloat16):
            all_text_feats = (
                text_cache.encode_with_grad(all_texts)
                if all_texts
                else torch.zeros(0, device=device)
            )

            patch_emb_s, _, patch_idx = student.forward_patches(points)
            patch_feat_s = proj(patch_emb_s)

            with torch.no_grad():
                if teacher_emb_cache is not None:
                    embs_t = torch.stack(
                        [teacher_emb_cache[_s["item_id"]].to(device) for _s in batch],
                        dim=0,
                    )
                else:
                    embs_t, _, _ = teacher.forward_patches(points)
                patch_feat_t = proj(embs_t)

            scale = temp_module()
            batch_loss_bce  = torch.tensor(0.0, device=device)
            batch_loss_kd_t = torch.tensor(0.0, device=device)
            valid_count = 0

            for b_idx, sample in enumerate(batch):
                _si, _ei = text_slices[b_idx]
                names_b = sample.get("label_names", [])
                if not names_b:
                    continue

                text_feats_b = all_text_feats[_si:_ei]
                if text_feats_b.numel() == 0:
                    continue

                lm = sample["label_masks"].to(device)
                loss_bce_b, Y_patch = dental_bce_loss(
                    patch_feat_s[b_idx], text_feats_b, lm, patch_idx[b_idx], scale
                )
                batch_loss_bce = batch_loss_bce + loss_bce_b
                valid_count += 1

                if lambda4 > 0.0:
                    with torch.no_grad():
                        logits_t = patch_feat_t[b_idx] @ text_feats_b.detach().T * scale
                    logits_s = patch_feat_s[b_idx] @ text_feats_b.T * scale
                    kl = kd_text_kl(logits_s, logits_t, temperature=4.0)
                    batch_loss_kd_t = batch_loss_kd_t + kl

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

            kd_loss = kd_patch_mse(patch_feat_s, patch_feat_t)
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
    parser = argparse.ArgumentParser("Stage 3b: 局所パッチ BCE + KD + EWC 学習 (ViT-bigG-14)")
    parser.add_argument("--config",    type=str, default="configs/stage3b_bigG14.yaml")
    parser.add_argument("--init_ckpt", type=str, default="",
                        help="初期化チェックポイント (Stage 2 patchalign3d.pt など)。"
                             "省略時は config の stage2_ckpt を使用。")
    parser.add_argument("--gpu",       type=str, default=None)
    parser.add_argument("--resume",    type=str, default="")
    args = parser.parse_args()

    cfg = load_config(args.config)

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg.get("seed", 42))

    out_dir = Path(cfg.get("output_dir", "outputs/stage3b_bigG14"))
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
    logger = logging.getLogger("stage3b_bigG14")
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
        text_augment  = False,
        point_augment = False,
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
    clip_model, text_cache = build_clip_and_cache(
        clip_model_name = clip_cfg.get("model", "ViT-bigG-14"),
        clip_pretrained = clip_cfg.get("pretrained", "laion2b_s39b_b160k"),
        device          = device,
        capacity        = clip_cfg.get("cache_capacity", 10000),
    )

    # ---- テキストエンコーダの最終層を解凍 (Stage 3b text_tune) ----
    text_tune_cfg = cfg.get("text_tune", {})
    text_tune_enabled = text_tune_cfg.get("enabled", False)
    if text_tune_enabled:
        n_last = text_tune_cfg.get("n_last_layers", 1)
        freeze_text_encoder_except_last(clip_model, n_last=n_last)
        _set_vit_text_train_mode(clip_model, training=True)
    else:
        logger.info("[text_tune] 無効 — テキストエンコーダは凍結のまま")

    # ---- モデル ----
    init_ckpt = args.init_ckpt or cfg.model.get("stage2_ckpt", "")
    student, teacher, proj, temp_module = build_stage3_models(
        cfg       = cfg.model,
        device    = device,
        ckpt_path = init_ckpt,
    )
    logger.info(f"モデルを初期化: init_ckpt={init_ckpt or '(なし)'}")

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
        ewc.to_device(device)

    # ---- Teacher patch_emb キャッシュ (frozen teacher は出力が不変) ----
    logger.info("Teacher patch_emb を事前計算中...")
    teacher_emb_cache: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for _sample in train_ds:
            _item_id = _sample["item_id"]
            if _item_id not in teacher_emb_cache:
                _pts = _sample["points"].unsqueeze(0).to(device)
                _emb, _, _ = teacher.forward_patches(_pts)
                teacher_emb_cache[_item_id] = _emb.squeeze(0).cpu()
    logger.info(f"Teacher キャッシュ完了: {len(teacher_emb_cache)} 患者")

    # ---- オプティマイザ ----
    point_params = (
        list(filter(lambda p: p.requires_grad, student.parameters()))
        + list(proj.parameters())
        + list(temp_module.parameters())
    )
    param_groups = [{"params": point_params, "lr": tr_cfg.get("lr", 3e-5)}]

    if text_tune_enabled:
        text_lr = text_tune_cfg.get("lr", 1e-5)
        # visual encoder のパラメータを除外してテキスト系のみ収集
        visual_ids = {id(p) for p in clip_model.visual.parameters()}
        text_params = [
            p for p in clip_model.parameters()
            if p.requires_grad and id(p) not in visual_ids
        ]
        param_groups.append({"params": text_params, "lr": text_lr})
        logger.info(
            f"[text_tune] テキストエンコーダ学習可能 {sum(p.numel() for p in text_params):,} params  lr={text_lr}"
        )

    optimizer = torch.optim.AdamW(
        param_groups,
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
        if text_tune_enabled and "text_encoder" in ckpt:
            _load_vit_text_encoder_state(clip_model, ckpt["text_encoder"])
        start_epoch = ckpt["epoch"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        logger.info(f"チェックポイントを再開: epoch={start_epoch-1}")

    # ---- 学習ループ ----
    eval_every = tr_cfg.get("eval_every", 5)
    save_every = tr_cfg.get("save_every", 10)
    global _copy_thread

    for epoch in range(start_epoch, n_epochs + 1):
        train_metrics = train_one_epoch(
            student, teacher, proj, temp_module, text_cache, ewc,
            train_loader, optimizer, device, cfg, epoch,
            clip_model=clip_model if text_tune_enabled else None,
            teacher_emb_cache=teacher_emb_cache,
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

        if epoch % eval_every == 0 or epoch == n_epochs:
            if text_tune_enabled:
                _set_vit_text_train_mode(clip_model, training=False)
                text_cache._store.clear()  # 重みが更新されたのでキャッシュを無効化
            val_metrics = eval_one_epoch(
                student, proj, temp_module, text_cache,
                val_loader, device, epoch
            )
            if text_tune_enabled:
                _set_vit_text_train_mode(clip_model, training=True)
            val_loss = val_metrics["val_loss"]
            logger.info(
                f"  -> val_loss={val_loss:.4f}"
                f"  val_patch_acc={val_metrics['patch_acc']:.3f}"
            )

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            # resume 用フル状態 (optimizer/scheduler 含む) → /tmp 経由で非同期コピー
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
            if text_tune_enabled:
                state["text_encoder"] = _save_vit_text_encoder_state(clip_model)
            _copy_thread = _save_with_local_buffer(state, ckpt_dir / "last.pt", "last")

            if is_best:
                # 推論用軽量版 (optimizer/scheduler を除外して ~170MB に削減)
                infer_state = {
                    "epoch":       epoch,
                    "student":     state["student"],
                    "proj":        state["proj"],
                    "val_metrics": val_metrics,
                }
                if text_tune_enabled:
                    infer_state["text_encoder"] = state["text_encoder"]
                _copy_thread = _save_with_local_buffer(infer_state, ckpt_dir / "best.pt", "best")
                logger.info(f"  -> Best チェックポイントを保存 (val_loss={val_loss:.4f})")

        if epoch % save_every == 0:
            periodic = {"epoch": epoch, "student": student.state_dict(), "proj": proj.state_dict()}
            if text_tune_enabled:
                periodic["text_encoder"] = _save_vit_text_encoder_state(clip_model)
            _copy_thread = _save_with_local_buffer(
                periodic, ckpt_dir / f"epoch_{epoch:03d}.pt", f"epoch_{epoch:03d}"
            )

    # ---- 最終チェックポイント ----
    # バックグラウンドコピーが完了してから best.pt を読む
    if _copy_thread is not None and _copy_thread.is_alive():
        _copy_thread.join()

    final_path = out_dir / "stage3b_best.pt"
    best_ckpt = torch.load(str(ckpt_dir / "best.pt"), map_location="cpu", weights_only=False)
    final_state = {
        "student": best_ckpt["student"],
        "proj":    best_ckpt["proj"],
        "epoch":   best_ckpt["epoch"],
    }
    if text_tune_enabled and "text_encoder" in best_ckpt:
        final_state["text_encoder"] = best_ckpt["text_encoder"]
    torch.save(final_state, final_path)
    logger.info(f"Stage 3b 完了。最良チェックポイント: {final_path}")


if __name__ == "__main__":
    main()
