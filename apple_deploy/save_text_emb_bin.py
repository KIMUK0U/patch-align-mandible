"""Convert text_embeddings.npz → text_embeddings.bin for Swift bundle.

Binary format read by DentalInference.swift loadTextEmbeddings():
  [Int32 LE] K  — number of labels
  [Int32 LE] D  — embedding dim (1280)
  [K * D * float32 LE] — embeddings row-major
  [UTF-8 newline-separated] — label strings

Usage:
    python precompute_text_emb.py --out text_embeddings.npz   # step 1 if not done
    python save_text_emb_bin.py --npz text_embeddings.npz --out text_embeddings.bin
    cp text_embeddings.bin DentalInferenceKit/Sources/DentalInferenceKit/Resources/
"""
import argparse
import struct
import numpy as np
from pathlib import Path


def convert(npz_path: str, out_path: str) -> None:
    td   = np.load(npz_path, allow_pickle=True)
    emb  = td["embeddings"].astype("<f4")          # float32 little-endian
    lbls = [str(l) for l in td["labels"]]

    K, D = emb.shape
    with open(out_path, "wb") as f:
        f.write(struct.pack("<ii", K, D))
        f.write(emb.tobytes())
        f.write("\n".join(lbls).encode("utf-8"))

    size_kb = Path(out_path).stat().st_size / 1024
    print(f"Saved K={K} labels, D={D} → {out_path}  ({size_kb:.1f} KB)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", default="outputs/npz/text_embeddings.npz")
    parser.add_argument("--out", default="text_embeddings.bin")
    args = parser.parse_args()
    convert(args.npz, args.out)
