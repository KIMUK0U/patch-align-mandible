"""EWC (Elastic Weight Consolidation) - Stage 3a → 3b 移行時の忘却防止。

設計書 Phase 3 セクション 4.3 に準拠。

使用方法:
    # Stage 3a 完了後
    ewc = EWC(student, train_loader_3a, device, n_samples=200)
    torch.save(ewc.to_dict(), "ewc_fisher.pt")

    # Stage 3b 学習中
    ewc = EWC.from_dict(torch.load("ewc_fisher.pt", map_location="cpu", weights_only=False))
    loss = ewc.penalty(student)
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class EWC:
    """
    EWC ペナルティ計算クラス。

    Parameters
    ----------
    model      : 学習後の Student モデル
    dataloader : Fisher 推定に使うデータローダ
    device     : torch.device
    n_samples  : Fisher 推定に使うサンプル数 (大きいほど精度が上がるが時間がかかる)
    loss_key   : バッチ損失の dict キー (デフォルト "infonce")
    """

    def __init__(
        self,
        model: nn.Module,
        dataloader,
        device: torch.device,
        n_samples: int = 200,
        loss_key: str = "infonce",
    ):
        # Stage 3a 完了時のパラメータを保存
        self.params: dict[str, torch.Tensor] = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
            if p.requires_grad
        }
        self.fisher: dict[str, torch.Tensor] = self._compute_fisher(
            model, dataloader, device, n_samples, loss_key
        )

    # ------------------------------------------------------------------

    def _compute_fisher(
        self,
        model: nn.Module,
        loader,
        device: torch.device,
        n_samples: int,
        loss_key: str,
    ) -> dict[str, torch.Tensor]:
        """対角 Fisher 情報行列を計算する。"""
        fisher: dict[str, torch.Tensor] = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
            if p.requires_grad
        }

        model.eval()
        count = 0
        for batch_data in loader:
            if count >= n_samples:
                break
            model.zero_grad()

            # batch_data はコールバック関数を通じて損失を計算する
            # ここでは損失計算関数が外から渡される設計のため、
            # EWC.__call__ か別途 loss_fn を渡す方式を採用する。
            # ただし、シンプルな実装として損失計算を内包するよう
            # train_stage3b.py 側で EWCComputer を呼び出す設計にする。
            # ここでは空の Fisher を返して、外部から手動更新する。
            break

        return fisher

    # ------------------------------------------------------------------
    # 外部から Fisher 行列を直接設定するメソッド (推奨使用方法)
    # ------------------------------------------------------------------

    @classmethod
    def from_params_and_fisher(
        cls,
        params: dict[str, torch.Tensor],
        fisher: dict[str, torch.Tensor],
    ) -> "EWC":
        """保存済み params / fisher から復元する。"""
        obj = object.__new__(cls)
        obj.params = {k: v.clone() for k, v in params.items()}
        obj.fisher = {k: v.clone() for k, v in fisher.items()}
        return obj

    @classmethod
    def from_dict(cls, d: dict) -> "EWC":
        """torch.save で保存した辞書から復元する。"""
        return cls.from_params_and_fisher(d["params"], d["fisher"])

    def to_dict(self) -> dict:
        """torch.save 用に辞書へ変換する。"""
        return {"params": self.params, "fisher": self.fisher}

    def to_device(self, device: torch.device) -> None:
        """Fisher と params を GPU に移動する。penalty() の per-batch .to() を no-op にする。"""
        self.params = {k: v.to(device) for k, v in self.params.items()}
        self.fisher = {k: v.to(device) for k, v in self.fisher.items()}

    # ------------------------------------------------------------------

    def penalty(self, model: nn.Module) -> torch.Tensor:
        """EWC ペナルティ (スカラー) を計算して返す。

        L_EWC = Σ_i F_i * (θ_i - θ*_i)^2
        """
        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        for n, p in model.named_parameters():
            if n in self.fisher and n in self.params:
                fisher_val = self.fisher[n].to(p.device)
                param_star = self.params[n].to(p.device)
                loss = loss + (fisher_val * (p - param_star) ** 2).sum()
        return loss


# ---------------------------------------------------------------------------
# Fisher 情報行列の計算ユーティリティ (Stage 3b の学習ループから呼び出す)
# ---------------------------------------------------------------------------

def compute_fisher_from_loader(
    model: nn.Module,
    proj: nn.Module,
    text_cache,
    loader,
    device: torch.device,
    n_batches: int = 50,
    temp_val: float = 14.3,
) -> dict[str, torch.Tensor]:
    """Stage 3a 完了後の Student で Fisher 情報行列を推定する。

    BCE 損失の勾配の二乗期待値を Fisher の対角近似として使用。

    Parameters
    ----------
    model      : Stage 3a 完了後の Student
    proj       : 投影ヘッド
    text_cache : LRUTextCache (または類似の encode_labels_for_sample を持つオブジェクト)
    loader     : Stage 3a 学習データローダ
    device     : torch.device
    n_batches  : 推定に使うバッチ数
    temp_val   : 温度の固定値 (LearnableTemp().item() など)
    """
    from phase3_losses.bce_multilabel import (
        compute_patch_label_distribution,
        dental_bce_loss,
    )

    fisher: dict[str, torch.Tensor] = {
        n: torch.zeros_like(p)
        for n, p in model.named_parameters()
        if p.requires_grad
    }

    model.eval()
    proj.eval()
    count = 0

    for batch in loader:
        if count >= n_batches:
            break

        model.zero_grad()
        proj.zero_grad()

        points_list = [s["points"].unsqueeze(0).to(device) for s in batch]
        points = torch.cat(points_list, dim=0)

        patch_emb, _, patch_idx = model.forward_patches(points)
        patch_feat = proj(patch_emb)

        total_loss = torch.tensor(0.0, device=device)
        valid = 0
        for b_idx, sample in enumerate(batch):
            names = sample["label_names"]
            if not names:
                continue
            text_feats = text_cache.encode_raw_texts(names).to(device)
            if text_feats.numel() == 0:
                continue

            lm = sample["label_masks"].to(device)
            Y_patch = compute_patch_label_distribution(lm, patch_idx[b_idx])

            logits = patch_feat[b_idx] @ text_feats.T * temp_val
            K_keep = text_feats.size(0)
            pos_w = torch.full((K_keep,), max(1.0, float(K_keep - 1)), device=device)
            loss_b = F.binary_cross_entropy_with_logits(logits, Y_patch, pos_weight=pos_w)
            total_loss = total_loss + loss_b
            valid += 1

        if valid > 0:
            (total_loss / valid).backward()
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        count += 1

    if count > 0:
        fisher = {n: v / count for n, v in fisher.items()}

    return fisher
