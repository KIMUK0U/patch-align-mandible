"""点群データ拡張ユーティリティ (Phase 3 歯科点群専用)。

適用順序: 回転(SO3) → スケーリング → 平行移動 → ジッタリング（点群のみ）

回転・スケーリング・平行移動はパッチ中心にも同じ変換を適用する。
ジッタリングはパッチ中心に適用しない（最近傍割り当てのラベル一貫性を保持するため）。
"""

import random

import torch


def random_so3_rotation() -> torch.Tensor:
    """一様分布の SO(3) 回転行列を生成 (3, 3)。

    QR 分解によりランダム直交行列を生成し、行列式が -1 の場合（反射）を
    符号修正して det = +1 の純粋な回転行列にする。
    """
    M = torch.randn(3, 3)
    Q, _ = torch.linalg.qr(M)
    if torch.det(Q).item() < 0:
        Q[:, 0] = -Q[:, 0]
    return Q  # (3, 3)


def augment_dental(
    xyz: torch.Tensor,
    centers: torch.Tensor,
    scale_range: tuple = (0.9, 1.1),
    translate_range: float = 0.05,
    jitter_sigma: float = 0.005,
    jitter_clip: float = 0.02,
) -> tuple[torch.Tensor, torch.Tensor]:
    """歯科点群にデータ拡張を適用する。

    Args:
        xyz:             点群座標 (N, 3)、単位球正規化済み
        centers:         パッチ中心座標 (K, 3)、同じ正規化済み
        scale_range:     等方スケーリングの範囲 (min, max)
        translate_range: 平行移動の最大幅（単位球スケール）
        jitter_sigma:    ジッタリングのガウスノイズ標準偏差
        jitter_clip:     ジッタリングのクリップ幅

    Returns:
        aug_xyz (N, 3), aug_centers (K, 3)
    """
    # (1) SO(3) ランダム回転 — xyz と centers に同じ R を適用
    R = random_so3_rotation()
    xyz     = xyz @ R.T
    centers = centers @ R.T

    # (2) 等方スケーリング — xyz と centers に同じ係数を適用
    s = random.uniform(*scale_range)
    xyz     = xyz * s
    centers = centers * s

    # (3) 平行移動 — xyz と centers に同じベクトルを加算
    t = torch.empty(3).uniform_(-translate_range, translate_range)
    xyz     = xyz + t
    centers = centers + t

    # (4) ジッタリング — 点群のみ（パッチ中心には適用しない）
    noise = torch.randn_like(xyz) * jitter_sigma
    noise = noise.clamp(-jitter_clip, jitter_clip)
    xyz   = xyz + noise

    return xyz, centers
