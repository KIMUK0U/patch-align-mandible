"""Pre-compute CLIP text embeddings for anatomical label names.

Run once on a machine with open_clip installed. The output npz is then used by
inference_mac.py and can be bundled in the iOS/visionOS Swift package as a binary.

Requirements:
    pip install open_clip_torch torch

Usage:
    python precompute_text_emb.py --out text_embeddings.npz
    python precompute_text_emb.py --labels custom_labels.txt --out text_embeddings.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import open_clip

DEFAULT_LABELS = [
    "left mandibular condyle",
    "right coronoid process",
    "left gonial angle",
    "midline mandible body",
    "right mandible body",
    "inferior border of mandible",
    "left ramus of mandible",
    "right mandibular condyle",
    "left coronoid process",
]


def compute_text_embeddings(
    labels: list[str],
    model_name: str = "ViT-bigG-14",
    pretrained: str = "laion2b_s39b_b160k",
) -> np.ndarray:
    """Returns (K, D) float32 L2-normalised embeddings."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(model_name)

    with torch.no_grad():
        tokens = tokenizer(labels).to(device)
        emb    = model.encode_text(tokens)
        emb    = torch.nn.functional.normalize(emb, dim=-1)

    return emb.cpu().float().numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels",     default=None, help="Text file with one label per line")
    parser.add_argument("--out",        default="outputs/npz/text_embeddings.npz")
    parser.add_argument("--model",      default="ViT-bigG-14")
    parser.add_argument("--pretrained", default="laion2b_s39b_b160k")
    args = parser.parse_args()

    labels = (
        Path(args.labels).read_text().strip().splitlines()
        if args.labels else DEFAULT_LABELS
    )

    print(f"Computing embeddings for {len(labels)} labels via {args.model}...")
    emb = compute_text_embeddings(labels, args.model, args.pretrained)

    out = args.out if args.out.endswith(".npz") else args.out + ".npz"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, embeddings=emb, labels=np.array(labels, dtype=object))
    print(f"Saved embeddings {emb.shape} → {out}")


if __name__ == "__main__":
    main()
