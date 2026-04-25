"""Stage 3b 主損失: BCE multi-label。

設計書 Phase 3 セクション 4.2 および stage2.py:463-466 に準拠。

ソフトラベル Y_patch (G, K) は compute_patch_label_distribution で算出する。
"""

import torch
import torch.nn.functional as F


def compute_patch_label_distribution(
    label_masks: torch.Tensor,   # (K, N) bool
    patch_idx: torch.Tensor,     # (G, M) long
) -> torch.Tensor:
    """パッチ単位のソフトラベル分布を計算する。

    stage2.py の compute_patch_label_distribution と同一実装。

    Args:
        label_masks: (K, N) bool — K 個のラベルマスク
        patch_idx:   (G, M) long — G パッチ各 M 点のインデックス

    Returns:
        (G, K) float — 各パッチ内での各ラベルの占有率 [0, 1]
    """
    K, N = label_masks.shape
    G, M = patch_idx.shape
    idx = patch_idx.reshape(1, -1).expand(K, -1)          # (K, G*M)
    gathered = label_masks.float().gather(1, idx).view(K, G, M)
    return gathered.mean(dim=-1).T.contiguous()            # (G, K)


def dental_bce_loss(
    patch_feat_s: torch.Tensor,   # (G, D) — 1 サンプル
    text_feats: torch.Tensor,     # (K, D)
    label_masks: torch.Tensor,    # (K, N) bool
    patch_idx: torch.Tensor,      # (G, M) long
    scale: torch.Tensor | float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1 サンプル分の BCE multi-label 損失を計算する。

    Returns:
        (loss, Y_patch)
          loss    : スカラーテンソル
          Y_patch : (G, K) ソフトラベル (忘却度評価などに再利用可能)
    """
    Y_patch = compute_patch_label_distribution(label_masks, patch_idx)  # (G, K)
    logits = patch_feat_s @ text_feats.T * scale                        # (G, K)
    K_keep = text_feats.size(0)
    pos_w = torch.full(
        (K_keep,), max(1.0, float(K_keep - 1)), device=text_feats.device
    )
    loss = F.binary_cross_entropy_with_logits(logits, Y_patch, pos_weight=pos_w)
    return loss, Y_patch


def dental_bce_loss_batch(
    patch_feat_s: torch.Tensor,    # (B, G, D)
    patch_idx: torch.Tensor,       # (B, G, M)
    batch_samples: list[dict],     # 各要素は label_masks / label_names を持つ
    text_cache,                    # encode_raw_texts(names) -> (K, D)
    scale: torch.Tensor | float,
    device: torch.device,
) -> tuple[torch.Tensor, float]:
    """バッチ全体の BCE 損失 (per-sample 平均)。

    Returns:
        (total_loss, patch_acc)
    """
    B = patch_feat_s.shape[0]
    G = patch_feat_s.shape[1]
    total_loss = torch.tensor(0.0, device=device)
    valid_count = 0
    total_correct = 0
    total_patches = 0

    for b in range(B):
        sample = batch_samples[b]
        names = sample.get("label_names", [])
        if not names:
            continue

        text_feats = text_cache.encode_raw_texts(names).to(device)  # (K, D)
        if text_feats.numel() == 0:
            continue

        lm = sample["label_masks"].to(device)                        # (K, N)
        loss_b, Y_patch = dental_bce_loss(
            patch_feat_s[b], text_feats, lm, patch_idx[b], scale
        )
        total_loss = total_loss + loss_b
        valid_count += 1

        # パッチ精度 (argmax が point_labels と一致する割合)
        with torch.no_grad():
            logits = patch_feat_s[b] @ text_feats.T * scale         # (G, K)
            pred = logits.argmax(dim=-1)                             # (G,)
            point_lbl = sample.get("point_labels", None)
            if point_lbl is not None:
                pl = point_lbl.to(device)
                # パッチの多数決ラベル
                from phase3_losses.infonce import (
                    _majority_label_per_patch,
                )
                hard = _majority_label_per_patch(pl, patch_idx[b], len(names))
                valid_p = hard >= 0
                if valid_p.any():
                    total_correct += (pred[valid_p] == hard[valid_p]).sum().item()
                    total_patches += valid_p.sum().item()

    if valid_count == 0:
        return torch.tensor(0.0, device=device, requires_grad=True), 0.0

    patch_acc = total_correct / max(total_patches, 1)
    return total_loss / valid_count, patch_acc
