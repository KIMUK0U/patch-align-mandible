"""FPS + KNN grouping — pure numpy, no CUDA / pointnet2_ops / knn_cuda."""
import numpy as np


def fps(xyz: np.ndarray, num_group: int) -> tuple[np.ndarray, np.ndarray]:
    """Furthest Point Sampling.

    Args:
        xyz       : (N, 3) float32
        num_group : G — number of centroids

    Returns:
        centers : (G, 3)
        indices : (G,) integer indices into xyz
    """
    N = len(xyz)
    selected = np.zeros(num_group, dtype=np.int32)
    dist = np.full(N, np.inf, dtype=np.float32)
    cur = 0
    for i in range(num_group):
        selected[i] = cur
        d = ((xyz - xyz[cur]) ** 2).sum(axis=1)
        np.minimum(dist, d, out=dist)
        cur = int(dist.argmax())
    return xyz[selected], selected


def knn(xyz: np.ndarray, centers: np.ndarray, k: int) -> np.ndarray:
    """Brute-force K-Nearest Neighbours.

    Args:
        xyz     : (N, 3)
        centers : (G, 3)
        k       : number of neighbours

    Returns:
        idx : (G, k) indices into xyz
    """
    diff  = xyz[None] - centers[:, None]       # (G, N, 3)
    dist2 = (diff ** 2).sum(axis=2)            # (G, N)
    return np.argsort(dist2, axis=1)[:, :k]    # (G, k)


def group_points(
    xyz: np.ndarray,
    num_group: int,
    group_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Group xyz into local patches via FPS + KNN.

    Args:
        xyz        : (N, 3) float32
        num_group  : G — number of patches
        group_size : M — points per patch

    Returns:
        neighborhood : (G, M, 3) relative coordinates
        centers      : (G, 3)
        patch_idx    : (G, M) absolute indices into xyz
    """
    centers, _   = fps(xyz, num_group)
    patch_idx    = knn(xyz, centers, group_size)           # (G, M)
    neighborhood = xyz[patch_idx] - centers[:, None, :]    # (G, M, 3)
    return neighborhood.astype(np.float32), centers.astype(np.float32), patch_idx
