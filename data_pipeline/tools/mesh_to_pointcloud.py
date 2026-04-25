"""
mesh_to_pointcloud.py — 下顎骨 PLY メッシュ → 点群変換

入力: 下顎骨全体の PLY ファイル (セグメント済みではない)
出力:
  - points_xyz:   (N, 3) float32  正規化済み LPS 点群
  - patch_centers: (K, 3) float32  FPS で選んだパッチ代表点

使い方:
    from tools.mesh_to_pointcloud import load_mandible_pointcloud, sample_patch_centers
"""

import sys
from pathlib import Path

import numpy as np
import open3d as o3d

# プロジェクトルートを sys.path に追加 (tools/ から import config できるように)
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools.lps_utils import normalize_pointcloud, fps_numpy


def load_mandible_pointcloud(
    ply_path: str | Path,
    n_sample: int = 8192,
) -> np.ndarray:
    """
    PLY メッシュファイルを読み込み、表面をサンプリングして正規化済み点群を返す。

    Args:
        ply_path: PLY ファイルパス
        n_sample: 初期サンプリング点数

    Returns:
        points_xyz: (n_sample, 3) float32  正規化済み LPS 座標
    """
    ply_path = Path(ply_path)
    if not ply_path.exists():
        raise FileNotFoundError(f"PLY not found: {ply_path}")

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_vertices():
        raise ValueError(f"Empty mesh: {ply_path}")

    mesh.compute_vertex_normals()

    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        # 三角形がない場合 (点群 PLY) はそのまま頂点を使う
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        if len(pts) > n_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), n_sample, replace=False)
            pts = pts[idx]

    pts = normalize_pointcloud(pts)
    return pts


def sample_patch_centers(
    points_xyz: np.ndarray,
    n_patches: int = 32,
    seed: int = 42,
) -> np.ndarray:
    """
    点群から FPS で n_patches 個のパッチ代表点を選ぶ。

    Args:
        points_xyz: (N, 3) float32
        n_patches:  選ぶパッチ数
        seed:       乱数シード

    Returns:
        patch_centers: (n_patches, 3) float32
        patch_indices: (n_patches,)  int64  — 元点群のインデックス
    """
    rng = np.random.default_rng(seed)
    idx = fps_numpy(points_xyz, n_patches, rng)
    return points_xyz[idx], idx


def downsample_to_final(
    points_xyz: np.ndarray,
    n_final: int = 2048,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    全点群を FPS で n_final 点にダウンサンプリング。

    Returns:
        pts_down:  (n_final, 3) float32
        down_idx:  (n_final,)   int64   — 元インデックス
    """
    rng = np.random.default_rng(seed)
    idx = fps_numpy(points_xyz, n_final, rng)
    return points_xyz[idx], idx


def build_patch_masks(
    points_xyz: np.ndarray,
    patch_centers: np.ndarray,
    radius: float = 0.15,
) -> np.ndarray:
    """
    各パッチ中心から radius 以内の点を True とするブールマスクを生成。

    Args:
        points_xyz:    (N, 3) float32  ダウンサンプリング済み点群
        patch_centers: (K, 3) float32
        radius:        パッチ半径 (正規化座標)

    Returns:
        masks: (K, N) bool
    """
    K, N = len(patch_centers), len(points_xyz)
    # (K, N) のブロードキャスト距離計算
    diff  = points_xyz[None, :, :] - patch_centers[:, None, :]  # (K, N, 3)
    dists = np.sqrt((diff ** 2).sum(axis=-1))                    # (K, N)
    return dists <= radius


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="PLY → 点群変換テスト")
    parser.add_argument("ply", help="PLY ファイルパス")
    parser.add_argument("--n_sample",  type=int, default=8192)
    parser.add_argument("--n_patches", type=int, default=32)
    parser.add_argument("--n_final",   type=int, default=2048)
    parser.add_argument("--radius",    type=float, default=0.15)
    args = parser.parse_args()

    print(f"Loading: {args.ply}")
    pts = load_mandible_pointcloud(args.ply, args.n_sample)
    print(f"  Loaded: {pts.shape}, range x=[{pts[:,0].min():.3f},{pts[:,0].max():.3f}]")

    centers, cidx = sample_patch_centers(pts, args.n_patches)
    print(f"  Patch centers: {centers.shape}")

    pts_down, didx = downsample_to_final(pts, args.n_final)
    print(f"  Downsampled:   {pts_down.shape}")

    # パッチ中心をダウンサンプリング後の座標系に対して計算
    masks = build_patch_masks(pts_down, centers, args.radius)
    print(f"  Masks: {masks.shape}, avg points/patch: {masks.sum(axis=1).mean():.1f}")
