"""Stage 3a 主損失: InfoNCE (cross-entropy 対照損失)。

設計書 Phase 3 セクション 3.3 に準拠。

patch_feat_s : (B, G, D) - Student パッチ特徴 (正規化済み)
text_feats   : (K, D)    - テキスト埋め込み (CLIP 空間)
hard_labels  : (B, G)    - 各パッチの領域 ID
scale        : スカラー   - 温度の逆数
"""

import torch
import torch.nn.functional as F


def infonce_loss_per_sample(
    patch_feat_s: torch.Tensor,     # (G, D)
    text_feats: torch.Tensor,       # (K, D)
    hard_labels: torch.Tensor,      # (G,) 値域 [0, K-1]、-1 は無効
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """1 サンプル分の InfoNCE 損失を計算する。

    有効なパッチ (hard_labels >= 0) のみ損失に含める。

    Returns: スカラーテンソル
    """
    valid = hard_labels >= 0
    if valid.sum() == 0:
        return torch.tensor(0.0, device=patch_feat_s.device, requires_grad=True)

    logits = patch_feat_s[valid] @ text_feats.T * scale   # (G_valid, K)
    targets = hard_labels[valid]                           # (G_valid,)
    return F.cross_entropy(logits, targets)


def infonce_loss_batch(
    patch_feat_s: torch.Tensor,     # (B, G, D)
    text_feats: torch.Tensor,       # (K, D)
    point_labels_b: list,           # list of B × (N,) point_labels Tensor
    patch_idx: torch.Tensor,        # (B, G, M) patch membership indices
    num_labels: int,
    scale: torch.Tensor | float,
) -> torch.Tensor:
    """バッチ全体の InfoNCE 損失 (per-sample 平均)。

    Parameters
    ----------
    patch_feat_s    : Student パッチ特徴 (B, G, D)
    text_feats      : CLIP テキスト特徴 (K, D)
    point_labels_b  : 各サンプルの per-point 領域 ID リスト
    patch_idx       : (B, G, M) パッチ所属点インデックス
    num_labels      : 領域数 K
    scale           : 温度スケール
    """
    B = patch_feat_s.shape[0]
    total_loss = torch.tensor(0.0, device=patch_feat_s.device)
    valid_count = 0

    for b in range(B):
        point_lbl = point_labels_b[b]           # (N,)
        if point_lbl is None:
            continue
        hard = _majority_label_per_patch(
            point_lbl, patch_idx[b], num_labels
        )                                        # (G,)
        loss_b = infonce_loss_per_sample(
            patch_feat_s[b], text_feats, hard, scale
        )
        total_loss = total_loss + loss_b
        valid_count += 1

    if valid_count == 0:
        return torch.tensor(0.0, device=patch_feat_s.device, requires_grad=True)

    return total_loss / valid_count


def _majority_label_per_patch(
    point_labels: torch.Tensor,   # (N,)
    patch_idx: torch.Tensor,      # (G, M)
    num_labels: int,
) -> torch.Tensor:
    """各パッチの多数決ラベルを求める。-1 は背景。

    Returns: (G,) long、有効パッチは [0, num_labels-1]、背景は -1
    """
    G, M = patch_idx.shape
    gathered = point_labels.gather(0, patch_idx.reshape(-1)).view(G, M)  # (G, M)

    # ラベル +1 してゼロパディング → one-hot count
    lbl_shifted = (gathered + 1).clamp_min(0)
    K = num_labels + 1
    one_hot = F.one_hot(lbl_shifted.clamp_max(K - 1), num_classes=K).sum(dim=1)
    one_hot[:, 0] = 0                   # 背景 (元の -1) をカウント対象外に

    has_any = one_hot.sum(dim=1) > 0
    preds = one_hot.argmax(dim=1) - 1   # 0-indexed に戻す

    out = torch.full((G,), -1, dtype=torch.long, device=point_labels.device)
    out[has_any] = preds[has_any]
    return out
