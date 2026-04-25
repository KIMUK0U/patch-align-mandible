"""Interactive patch search: type free text, find similar dental patches on Mac.

Pipeline:
  1. PLY → FPS+KNN → MLX point encoder → patch embeddings (G, 1280)
  2. CLIP text encoder (MLX, Apple GPU via Metal) → text embedding (1280,)
  3. Cosine similarity → ranked top-K patches

Usage:
    cd Phase3_MLX
    python clip_text/search_mac.py \
        --ply scan.ply \
        --patch_weights phase3_weights.npz \
        --clip_weights  clip_text/clip_text_weights.npz \
        --topk 5
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))   # Phase3_MLX/
from model import PointTransformerMLX, group_points
from clip_text.mlx_clip_text import CLIPTextMLX

import mlx.core as mx


def _load_ply(path: str) -> np.ndarray:
    try:
        from plyfile import PlyData
    except ImportError:
        raise ImportError("pip install plyfile")
    v = PlyData.read(path)["vertex"]
    return np.stack([v["x"], v["y"], v["z"]], axis=1).astype(np.float32)


def _normalize(xyz: np.ndarray) -> np.ndarray:
    xyz = xyz - xyz.mean(axis=0)
    return (xyz / (np.abs(xyz).max() + 1e-8)).astype(np.float32)


def compute_patch_embeddings(
    ply_path: str,
    patch_weights: str,
    num_group: int = 128,
    group_size: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Run point encoder on PLY → (patch_emb (G,1280), centers (G,3))."""
    xyz = _normalize(_load_ply(ply_path))
    neighborhood, centers, _ = group_points(xyz, num_group, group_size)
    model = PointTransformerMLX.from_npz(patch_weights)
    emb   = model(mx.array(neighborhood), mx.array(centers))
    mx.eval(emb)
    return np.array(emb), centers


def search_patches(
    query: str,
    patch_emb: np.ndarray,   # (G, 1280) L2-normalised
    centers: np.ndarray,      # (G, 3)
    clip_enc: CLIPTextMLX,
    topk: int = 5,
) -> list[dict]:
    """Encode query text and rank patches by cosine similarity."""
    text_emb = np.array(clip_enc.encode([query]))[0]    # (1280,)
    scores   = patch_emb @ text_emb                     # (G,)
    top_idx  = np.argsort(scores)[::-1][:topk]
    return [
        {
            "rank":   rank + 1,
            "patch":  int(i),
            "score":  float(scores[i]),
            "center": centers[i].tolist(),
        }
        for rank, i in enumerate(top_idx)
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ply",           required=True,           help="Input PLY file")
    parser.add_argument("--patch_weights", required=True,           help="phase3_weights.npz")
    parser.add_argument("--clip_weights",  required=True,           help="clip_text_weights.npz")
    parser.add_argument("--topk",          type=int, default=5)
    parser.add_argument("--num_group",     type=int, default=128)
    parser.add_argument("--group_size",    type=int, default=32)
    args = parser.parse_args()

    print("Computing patch embeddings from PLY (this may take ~10 s)...")
    patch_emb, centers = compute_patch_embeddings(
        args.ply, args.patch_weights, args.num_group, args.group_size
    )
    print(f"  {len(patch_emb)} patch embeddings ready")

    print("Loading CLIP text encoder (MLX)...")
    clip_enc = CLIPTextMLX.from_npz(args.clip_weights)
    print("  Ready.\n")
    print("Type an anatomical description to search for similar patches.")
    print("Examples: 'condyle', 'ramus', 'inferior border', 'gonial angle'")
    print("Press Ctrl-C or Ctrl-D to quit.\n")

    while True:
        try:
            query = input("Query> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            break
        if not query:
            continue

        results = search_patches(query, patch_emb, centers, clip_enc, args.topk)
        print(f"\nTop {args.topk} results for: \"{query}\"")
        for r in results:
            cx, cy, cz = r["center"]
            print(f"  #{r['rank']}  patch {r['patch']:3d}  "
                  f"score={r['score']:.4f}  "
                  f"center=({cx:+.3f}, {cy:+.3f}, {cz:+.3f})")
        print()


if __name__ == "__main__":
    main()
