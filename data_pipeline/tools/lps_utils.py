"""
lps_utils.py — LPS 座標系ユーティリティ

医療画像標準 LPS 座標系:
  x 軸: + = Left (患者の左)   / − = Right
  y 軸: + = Posterior (後方) / − = Anterior (前方)
  z 軸: + = Superior (上方)  / − = Inferior (下方)

正中判定:
  |x| <= MIDLINE_THRESHOLD の場合は左右ではなく正中 (M-) オクタントを返す。
  下顎骨の正規化座標で x ≈ ±0.12 以内は交連部・正中付近に相当する。
"""

import numpy as np

# 正中と判定する x 座標絶対値の閾値 (正規化座標)
MIDLINE_THRESHOLD = 0.12


# 12 オクタント (8 側方 + 4 正中) と代表的な下顎骨解剖部位
OCTANT_ANATOMY_HINTS = {
    "L-A-S": "left coronoid process, left anterior superior mandible",
    "L-A-I": "left anterior inferior mandible, left mental foramen region",
    "L-P-S": "left mandibular condyle, left condylar neck",
    "L-P-I": "left mandibular ramus inferior, left mandibular angle",
    "R-A-S": "right coronoid process, right anterior superior mandible",
    "R-A-I": "right anterior inferior mandible, right mental foramen region",
    "R-P-S": "right mandibular condyle, right condylar neck",
    "R-P-I": "right mandibular ramus inferior, right mandibular angle",
    # 正中オクタント
    "M-A-S": "symphysis menti superior, central incisor alveolar process, midline anterior",
    "M-A-I": "symphysis menti, mental protuberance, mental tubercle, chin midline",
    "M-P-S": "lingual midline superior, central incisor lingual plate, sublingual midline",
    "M-P-I": "genial tubercle, mental spine, lingual midline inferior",
}

POSITION_DESCRIPTIONS = {
    "L-A-S": "left anterior superior region",
    "L-A-I": "left anterior inferior region",
    "L-P-S": "left posterior superior region",
    "L-P-I": "left posterior inferior region",
    "R-A-S": "right anterior superior region",
    "R-A-I": "right anterior inferior region",
    "R-P-S": "right posterior superior region",
    "R-P-I": "right posterior inferior region",
    # 正中オクタント
    "M-A-S": "midline anterior superior region",
    "M-A-I": "midline anterior inferior region",
    "M-P-S": "midline posterior superior region",
    "M-P-I": "midline posterior inferior region",
}


def lps_octant(center_lps: np.ndarray) -> str:
    """
    center_lps: (3,) [x=L, y=P, z=S] 正規化済み LPS 座標

    Returns:
        例 "L-P-S", "R-A-I", "M-A-I" (正中の場合は "M-" プレフィックス)
    """
    x, y, z = center_lps
    if abs(x) <= MIDLINE_THRESHOLD:
        lr = "M"
    else:
        lr = "L" if x > 0 else "R"
    pa = "P" if y >= 0 else "A"
    si = "S" if z >= 0 else "I"
    return f"{lr}-{pa}-{si}"


def is_midline(center_lps: np.ndarray) -> bool:
    """パッチ中心が正中帯 (|x| <= MIDLINE_THRESHOLD) にあるか判定"""
    return abs(center_lps[0]) <= MIDLINE_THRESHOLD


def octant_to_lr(octant: str) -> tuple[str, str]:
    """
    Returns: (lr_side, opposite_side)
        左右オクタント: ("left", "right") または ("right", "left")
        正中オクタント: ("midline", "")
    """
    if octant.startswith("M"):
        return "midline", ""
    lr_side  = "left"  if octant.startswith("L") else "right"
    opp_side = "right" if lr_side == "left"       else "left"
    return lr_side, opp_side


def normalize_pointcloud(pts: np.ndarray) -> np.ndarray:
    """
    センタリング + 単位球正規化 (color=False / C=3 専用)
    pts: (N, 3) float32
    """
    pts = pts - pts.mean(axis=0)
    scale = np.abs(pts).max() + 1e-9
    return pts / scale


def get_normalization_params(pts: np.ndarray) -> tuple[np.ndarray, float]:
    """
    normalize_pointcloud と同一の変換 (pts - mean) / scale の
    パラメータ (mean, scale) を返す。
    別ソース (STL等) に同じ正規化を適用したい場合に使用する。
    """
    mean  = pts.mean(axis=0)
    scale = float(np.abs(pts - mean).max()) + 1e-9
    return mean, scale


def apply_normalization(pts: np.ndarray, mean: np.ndarray, scale: float) -> np.ndarray:
    """get_normalization_params で得たパラメータで点群を正規化する"""
    return (pts - mean) / scale


def fps_numpy(xyz: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """
    Farthest Point Sampling (numpy 実装)
    xyz: (N, 3)
    k:   サンプル数
    Returns: (k,) インデックス配列
    """
    N = len(xyz)
    k = min(k, N)
    idx  = np.zeros(k, dtype=np.int64)
    dist = np.full(N, np.inf)
    idx[0] = rng.integers(0, N)
    for i in range(1, k):
        d    = np.sum((xyz - xyz[idx[i - 1]]) ** 2, axis=1)
        dist = np.minimum(dist, d)
        idx[i] = np.argmax(dist)
    return idx


def mask_aware_fps(
    points_xyz: np.ndarray,
    point_labels: np.ndarray,
    total: int = 2048,
    min_per_label: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """
    各ラベルから min_per_label 点を確保しつつ FPS でダウンサンプリング。
    point_labels に -1 (unlabeled) が含まれてもよい。

    Returns: (total,) インデックス配列
    """
    rng = np.random.default_rng(seed)
    unique_ids = [i for i in np.unique(point_labels) if i >= 0]
    chosen: set[int] = set()

    for lid in unique_ids:
        mask = np.where(point_labels == lid)[0]
        take = min(min_per_label, len(mask))
        sub  = fps_numpy(points_xyz[mask], take, rng)
        chosen.update(mask[sub].tolist())

    if len(chosen) < total:
        unchosen = np.array([i for i in range(len(points_xyz)) if i not in chosen])
        need  = total - len(chosen)
        extra = fps_numpy(points_xyz[unchosen], need, rng)
        chosen.update(unchosen[extra].tolist())

    return np.array(sorted(chosen)[:total], dtype=np.int64)
