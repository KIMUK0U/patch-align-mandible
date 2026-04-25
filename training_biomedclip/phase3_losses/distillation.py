"""知識蒸留 (KD) 損失。

設計書 Phase 3 セクション 3.3 (KD) と 4.4 (Stage 3b KD) に準拠。

MSE: Student パッチ特徴 ↔ Teacher パッチ特徴
KL : Student logits / Teacher logits (オプション)
"""

import torch
import torch.nn.functional as F


def kd_patch_mse(
    patch_feat_s: torch.Tensor,   # (B, G, D) or (G, D)
    patch_feat_t: torch.Tensor,   # same shape
) -> torch.Tensor:
    """Student と Teacher のパッチ特徴間 MSE 損失 (スカラー)。

    設計書:
        kd_loss = F.mse_loss(patch_feat_s, patch_feat_t.detach())
    """
    return F.mse_loss(patch_feat_s, patch_feat_t.detach())


def kd_text_kl(
    logits_s: torch.Tensor,       # (G, K)
    logits_t: torch.Tensor,       # (G, K)
    temperature: float = 4.0,
) -> torch.Tensor:
    """Teacher logits を soft target とした KL 蒸留損失 (スカラー)。

    設計書:
        F.kl_div(log_softmax(logits_s / T), softmax(logits_t / T), ...)
    """
    log_p_s = F.log_softmax(logits_s / temperature, dim=-1)
    p_t     = F.softmax(logits_t / temperature, dim=-1)
    return F.kl_div(log_p_s, p_t, reduction="batchmean") * (temperature ** 2)


def combined_kd_loss(
    patch_feat_s: torch.Tensor,      # (B, G, D)
    patch_feat_t: torch.Tensor,      # (B, G, D) detach 済み
    lambda_mse: float = 0.5,
    logits_s: torch.Tensor | None = None,   # (B*G, K) optional
    logits_t: torch.Tensor | None = None,   # (B*G, K) optional
    lambda_kl: float = 0.2,
    kl_temperature: float = 4.0,
) -> torch.Tensor:
    """MSE + KL (オプション) の合算 KD 損失。

    Parameters
    ----------
    patch_feat_s : Student パッチ特徴
    patch_feat_t : Teacher パッチ特徴 (detach されている前提)
    lambda_mse   : MSE 損失の重み
    logits_s     : Student logits (KL 用、None でスキップ)
    logits_t     : Teacher logits (KL 用、None でスキップ)
    lambda_kl    : KL 損失の重み
    kl_temperature : KL 蒸留温度
    """
    loss = lambda_mse * kd_patch_mse(patch_feat_s, patch_feat_t)

    if logits_s is not None and logits_t is not None:
        loss = loss + lambda_kl * kd_text_kl(logits_s, logits_t, kl_temperature)

    return loss
