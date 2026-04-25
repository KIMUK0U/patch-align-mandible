"""Extract CLIP ViT-bigG-14 text encoder weights → numpy npz.

Saves only the text encoder weights (no visual encoder) for use by
mlx_clip_text.py (Mac) and export_clip_coreml.py (iOS/visionOS).

Requirements:
    pip install open_clip_torch torch

Usage:
    python clip_text/extract_clip_weights.py
    python clip_text/extract_clip_weights.py --out clip_text_weights.npz
"""
import argparse
import numpy as np
import open_clip


def extract(model_name: str, pretrained: str, out_path: str) -> None:
    print(f"Loading {model_name} / {pretrained} ...")
    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    model.eval()

    sd = model.state_dict()
    text_keys = [k for k in sd if not k.startswith("visual.")]
    print(f"Found {len(text_keys)} text encoder keys")

    arrays = {k: sd[k].float().numpy() for k in text_keys}

    out = out_path if out_path.endswith(".npz") else out_path + ".npz"
    np.savez(out, **arrays)
    print(f"Saved → {out}")
    for k in sorted(arrays)[:10]:
        print(f"  {k}: {arrays[k].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="ViT-bigG-14")
    parser.add_argument("--pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--out",        default="clip_text_weights.npz")
    args = parser.parse_args()
    extract(args.model, args.pretrained, args.out)
