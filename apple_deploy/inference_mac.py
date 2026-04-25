"""Mac inference with MLX — no PyTorch / CUDA required.

Prerequisites:
    pip install mlx numpy plyfile

Step 1 — Convert weights once:
    python convert_weights.py --ckpt ../Phase3_Training/outputs/stage3a/stage3a_last.pt

Step 2 — Pre-compute text embeddings once (needs open_clip):
    python precompute_text_emb.py --out text_embeddings.npz

Step 3 — Run inference:
    python inference_mac.py --weights phase3_weights.npz \
                            --ply scan.ply \
                            --text_emb text_embeddings.npz
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from model import PointTransformerMLX, group_points

import mlx.core as mx


# --------------------------------------------------------------------------
# PLY loader (plyfile only, no open3d)
# --------------------------------------------------------------------------

def load_ply(path: str) -> np.ndarray:
    """Load xyz from PLY file. Returns (N, 3) float32."""
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("pip install plyfile")
    ply = PlyData.read(path)
    v   = ply["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def normalize_point_cloud(xyz: np.ndarray) -> np.ndarray:
    """Center and scale to unit sphere."""
    xyz = xyz - xyz.mean(axis=0)
    xyz = xyz / (np.abs(xyz).max() + 1e-8)
    return xyz.astype(np.float32)


# --------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------

def run(
    weights_path: str,
    ply_path: str,
    text_emb_path: str,
    num_group: int  = 128,
    group_size: int = 32,
    out_json: str   = "results.json",
) -> list[dict]:
    # 1. Load and normalise point cloud
    xyz = normalize_point_cloud(load_ply(ply_path))
    print(f"Loaded {len(xyz)} points from {ply_path}")

    # 2. FPS + KNN grouping (pure numpy — no CUDA)
    neighborhood, centers, _ = group_points(xyz, num_group, group_size)
    print(f"Grouped into {num_group} patches x {group_size} pts")

    # 3. Load model weights
    model = PointTransformerMLX.from_npz(weights_path)

    # 4. Forward pass on Apple GPU via Metal
    patch_emb = model(mx.array(neighborhood), mx.array(centers))  # (G, 1280)
    mx.eval(patch_emb)
    patch_emb_np = np.array(patch_emb)

    # 5. Load text embeddings  (K, 1280) + label names (K,)
    td        = np.load(text_emb_path, allow_pickle=True)
    text_emb  = td["embeddings"].astype(np.float32)   # (K, 1280) L2-normalised
    labels    = td["labels"]                           # (K,) str

    # 6. Cosine similarity (both sides already L2-normalised)
    scores   = patch_emb_np @ text_emb.T              # (G, K)
    pred_idx = scores.argmax(axis=1)                  # (G,)

    # 7. Collate results
    results = [
        {
            "patch":      int(g),
            "label":      str(labels[pred_idx[g]]),
            "score":      float(scores[g, pred_idx[g]]),
            "center_xyz": centers[g].tolist(),
        }
        for g in range(num_group)
    ]
    results.sort(key=lambda r: -r["score"])

    print("\nTop-10 patch predictions:")
    for r in results[:10]:
        print(f"  patch {r['patch']:3d}  {r['label']:35s}  score={r['score']:.4f}")

    if out_json:
        Path(out_json).write_text(json.dumps(results, indent=2, ensure_ascii=False))
        print(f"\nSaved {len(results)} results → {out_json}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights",    required=True)
    parser.add_argument("--ply",        required=True)
    parser.add_argument("--text_emb",   required=True)
    parser.add_argument("--num_group",  type=int, default=128)
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--out_json",   default="results.json")
    args = parser.parse_args()
    run(args.weights, args.ply, args.text_emb, args.num_group, args.group_size, args.out_json)
