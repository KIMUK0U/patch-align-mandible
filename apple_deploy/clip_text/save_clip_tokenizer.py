"""Extract CLIP BPE tokenizer vocab and merge rules for Swift bundle.

Outputs:
  clip_vocab.json   — {"<token_string>": token_id, ...}  (49 408 entries)
  clip_merges.txt   — merge rules, one per line: "tok_a tok_b"

These files are bundled into DentalInferenceKit/Resources/ and loaded
by CLIPTokenizer.swift for on-device tokenisation on iOS/visionOS/macOS.

Requirements:
    pip install open_clip_torch

Usage:
    cd Phase3_MLX
    python clip_text/save_clip_tokenizer.py
    python clip_text/save_clip_tokenizer.py --out_dir clip_text/
"""
import argparse
import json
from pathlib import Path

import open_clip


def save(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

    # Unwrap to SimpleTokenizer
    tok = tokenizer
    if hasattr(tok, "tokenizer"):
        tok = tok.tokenizer
    if not hasattr(tok, "encoder"):
        raise RuntimeError(
            "Cannot locate encoder dict — inspect tokenizer attributes: "
            + str(dir(tok))
        )

    # vocab: token_string → int id
    vocab: dict[str, int] = dict(tok.encoder)
    vocab_path = out / "clip_vocab.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)
    print(f"Saved vocab  ({len(vocab)} tokens) → {vocab_path}")

    # merges: sorted by rank ascending → "tok_a tok_b" per line
    merges_sorted = sorted(tok.bpe_ranks.items(), key=lambda kv: kv[1])
    merge_lines = [f"{a} {b}" for (a, b), _ in merges_sorted]
    merges_path = out / "clip_merges.txt"
    merges_path.write_text("\n".join(merge_lines), encoding="utf-8")
    print(f"Saved merges ({len(merge_lines)} rules)  → {merges_path}")

    print("\nCopy both files to:")
    print("  DentalInferenceKit/Sources/DentalInferenceKit/Resources/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out_dir",
        default=".",
        help="Directory to write clip_vocab.json and clip_merges.txt",
    )
    args = parser.parse_args()
    save(args.out_dir)
