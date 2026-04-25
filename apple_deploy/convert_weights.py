"""Phase3 PyTorch checkpoint (.pt) → numpy npz

Usage:
    python convert_weights.py --ckpt ../Phase3_Training/outputs/stage3a/stage3a_last.pt
    python convert_weights.py --ckpt best.pt --out my_weights.npz
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def convert(ckpt_path: str, out_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    student_state = ckpt.get("student", {})
    proj_state    = ckpt.get("proj", {})
    temp_state    = ckpt.get("temp", {})

    arrays: dict[str, np.ndarray] = {}
    for k, v in student_state.items():
        arrays[f"student.{k}"] = v.float().numpy()
    for k, v in proj_state.items():
        arrays[f"proj.{k}"] = v.float().numpy()
    for k, v in temp_state.items():
        arrays[f"temp.{k}"] = v.float().numpy()

    out = Path(out_path)
    if out.suffix == ".npz":
        out = out.with_suffix("")
    np.savez(str(out), **arrays)
    print(f"Saved {len(arrays)} tensors → {out}.npz")
    print("Keys sample:", list(arrays.keys())[:6])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--out",  default="outputs/npz/phase3_weights", help="Output path (.npz auto-appended; directory is created automatically)")
    args = parser.parse_args()
    convert(args.ckpt, args.out)
