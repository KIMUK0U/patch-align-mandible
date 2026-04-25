"""
build_patchalign_dataset.py — verified_labels.json → PatchAlign3D 互換フォーマット変換

出力フォーマット (TrainingSetDataset 要求仕様):
    data/DentalPatchData/dental_dataset/labeled/
    ├── split/
    │   ├── train.txt
    │   └── val.txt
    ├── points/
    │   └── <uid>/
    │       └── points.pt          # (2048, 3) float32, 正規化済み
    └── rendered/
        └── dental_<case_id>/
            └── oriented/
                └── masks/
                    └── merged/
                        ├── mask2points.pt    # (L, 2048) bool
                        └── mask_labels.txt   # L 行 (simple ラベル)

実行:
    cd /path/to/ULIP_PointLLM
    python DentalPatchAligned3D/Phase0_Data/tools/build_patchalign_dataset.py \
        --val_ratio 0.2
"""

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

_PHASE0_DIR = Path(__file__).resolve().parent.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

import config as cfg
from tools.mesh_to_pointcloud import (
    load_mandible_pointcloud,
    downsample_to_final,
    build_patch_masks,
    sample_patch_centers,
)
from tools.lps_utils import mask_aware_fps


def build_dataset(
    verified_path: Path,
    ply_dir: Path,
    dataset_root: Path,
    split_dir: Path,
    n_final: int = 2048,
    patch_radius: float = 0.15,
    n_patches: int = 32,
    n_sample: int = 8192,
    val_ratio: float = 0.2,
    min_points_per_mask: int = 8,
    seed: int = 42,
) -> None:
    """
    verified_labels.json の各症例を PatchAlign3D 互換フォーマットに変換する。
    """
    with open(verified_path, encoding="utf-8") as f:
        verified: dict[str, dict] = json.load(f)

    ply_files = cfg.get_ply_files()

    # 採用済みエントリを症例 × パッチ IDでインデックス
    accepted_by_case: dict[str, list[dict]] = {}
    for entry in verified.values():
        if entry["status"] not in ("accepted", "edited", "all_accepted"):
            continue
        case_id  = entry["case_id"]
        if case_id not in accepted_by_case:
            accepted_by_case[case_id] = []
        accepted_by_case[case_id].append(entry)

    item_ids: list[str] = []

    for case_id, entries in accepted_by_case.items():
        if case_id not in ply_files:
            print(f"  [WARNING] PLY not found for {case_id}, skip")
            continue

        ply_path = ply_files[case_id]
        print(f"\n[{case_id}] n_accepted_patches={len(entries)}")

        # 点群読み込み・ダウンサンプリング
        pts_full = load_mandible_pointcloud(ply_path, n_sample)
        pts_down, _ = downsample_to_final(pts_full, n_final, seed=seed)

        # パッチ中心を復元 (候補 JSON の座標を使用)
        patch_centers = np.array(
            [e["all_candidates_center"] if "all_candidates_center" in e
             else e.get("patch_center_lps", [0, 0, 0])
             for e in entries],
            dtype=np.float32,
        )

        # patch_center_lps が entries に含まれているケースに対応
        # (verified_labels.json に patch_center_lps を保存していないパスへの対処)
        valid_entries = []
        valid_centers = []
        for e in entries:
            # candidates.json から中心座標を再取得する必要があるため
            # entry に patch_center_lps が含まれる前提で進む
            # (verify_text_labels.py が all_candidates に含む想定)
            center = _get_center(e, cfg.CANDIDATES_JSON)
            if center is not None:
                valid_entries.append(e)
                valid_centers.append(center)

        if not valid_centers:
            print(f"  [WARNING] パッチ中心座標が取得できませんでした: {case_id}")
            continue

        centers_np = np.array(valid_centers, dtype=np.float32)
        masks_full  = build_patch_masks(pts_down, centers_np, radius=patch_radius)  # (K, N_final)

        # 有効マスク (点数が min_points_per_mask 以上) のみ残す
        masks_out, labels_out, texts_out = [], [], []
        for i, (mask, entry) in enumerate(zip(masks_full, valid_entries)):
            if mask.sum() < min_points_per_mask:
                print(f"    Patch {i:03d}: mask too sparse ({mask.sum()} pts), skip")
                continue
            label = entry.get("mask_label") or entry.get("simple", "")
            if not label:
                continue
            # 採用済みテキストリスト (accepted_texts がなければ mask_label のみ)
            accepted = entry.get("accepted_texts") or [label]
            masks_out.append(mask)
            labels_out.append(label)
            texts_out.append([t for t in accepted if t])

        if not masks_out:
            print(f"  [WARNING] 有効マスクなし: {case_id}")
            continue

        # ── UID & 出力パス ──
        uid      = case_id.replace(" ", "_").replace("/", "_")
        item_id  = f"dental_{uid}"
        item_ids.append(item_id)

        pts_out_dir  = dataset_root / "points"  / uid
        mask_out_dir = dataset_root / "rendered" / item_id / "oriented" / "masks" / "merged"
        pts_out_dir.mkdir(parents=True, exist_ok=True)
        mask_out_dir.mkdir(parents=True, exist_ok=True)

        # points.pt
        pts_tensor = torch.from_numpy(pts_down).float()  # (2048, 3)
        torch.save(pts_tensor, pts_out_dir / "points.pt")

        # mask2points.pt
        masks_tensor = torch.tensor(np.stack(masks_out), dtype=torch.bool)  # (L, 2048)
        torch.save(masks_tensor, mask_out_dir / "mask2points.pt")

        # mask_labels.txt  — 1行1ラベル (canonical label = accepted_texts[0])
        (mask_out_dir / "mask_labels.txt").write_text(
            "\n".join(labels_out), encoding="utf-8"
        )

        # mask_texts.json  — マスクインデックス → 採用テキストリスト (多テキスト対応)
        mask_texts = {str(i): txts for i, txts in enumerate(texts_out)}
        with open(mask_out_dir / "mask_texts.json", "w", encoding="utf-8") as f:
            json.dump(mask_texts, f, ensure_ascii=False, indent=2)

        print(f"  Saved: {len(masks_out)} masks, points={pts_tensor.shape}, "
              f"masks={masks_tensor.shape}, "
              f"avg_texts={sum(len(t) for t in texts_out)/len(texts_out):.1f}")

    # ── train / val split ──
    random.seed(seed)
    random.shuffle(item_ids)
    n_val   = max(1, int(len(item_ids) * val_ratio))
    val_ids = item_ids[:n_val]
    trn_ids = item_ids[n_val:]

    split_dir.mkdir(parents=True, exist_ok=True)
    (split_dir / "train.txt").write_text("\n".join(trn_ids), encoding="utf-8")
    (split_dir / "val.txt").write_text(  "\n".join(val_ids),  encoding="utf-8")

    print(f"\nSplit: train={len(trn_ids)}, val={len(val_ids)}")
    print(f"Dataset saved to: {dataset_root}")


def _get_center(entry: dict, candidates_json: Path) -> np.ndarray | None:
    """
    entry から patch_center_lps を取得する。
    verified_labels.json には含まれていない場合があるため candidates.json を参照する。
    """
    # パターン 1: entry に直接含まれている
    if "patch_center_lps" in entry:
        return np.array(entry["patch_center_lps"], dtype=np.float32)

    # パターン 2: candidates.json を参照
    if candidates_json.exists():
        with open(candidates_json, encoding="utf-8") as f:
            candidates = json.load(f)
        case_id  = entry.get("case_id")
        patch_id = entry.get("patch_id")
        if case_id in candidates:
            for p in candidates[case_id]:
                if p["patch_id"] == patch_id and "patch_center_lps" in p:
                    return np.array(p["patch_center_lps"], dtype=np.float32)
    return None


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="PatchAlign3D 互換データセット構築")
    parser.add_argument("--val_ratio",    type=float, default=0.2)
    parser.add_argument("--n_final",      type=int,   default=2048)
    parser.add_argument("--patch_radius", type=float, default=0.15)
    parser.add_argument("--n_patches",    type=int,   default=32)
    parser.add_argument("--n_sample",     type=int,   default=8192)
    parser.add_argument("--seed",         type=int,   default=42)
    args = parser.parse_args()

    build_dataset(
        verified_path  = cfg.VERIFIED_JSON,
        ply_dir        = cfg.PLY_DIR,
        dataset_root   = cfg.DATASET_ROOT,
        split_dir      = cfg.SPLIT_DIR,
        n_final        = args.n_final,
        patch_radius   = args.patch_radius,
        n_patches      = args.n_patches,
        n_sample       = args.n_sample,
        val_ratio      = args.val_ratio,
        seed           = args.seed,
    )
