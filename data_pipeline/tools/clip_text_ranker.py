"""
clip_text_ranker.py — OpenCLIP ViT-bigG-14 による解剖ラベル候補ランキング

LLM API を使わずに、CLIP の画像–テキスト類似度 (cosine similarity) を用いて
クローズドボキャブラリから上位候補を提示する亜種モジュール。

【手法】
  z 軸周り 8 方位 (0°, 45°, 90°, …, 315°) のレンダリング画像を用いて
  画像特徴量を平均プーリング → テキスト特徴量との cosine 類似度でランキング。

【出力形式】
  candidates_clip.json: LLM モードと同一フォーマット (texts フィールド) に加え、
  clip_scores フィールドで各候補の CLIP スコアを保存する。

実行例 (run_phase0.py 経由):
    python run_phase0.py --mode clip --step 1 --device cuda
    python run_phase0.py --mode clip --step 2
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

def auto_device() -> str:
    """
    利用可能な最速デバイスを返す: cuda > mps > cpu

    MPS (Apple Silicon) 対応状況:
      - open_clip の encode_image / encode_text は標準 PyTorch 演算のみ使用するため
        MPS バックエンドで動作する (PyTorch >= 2.0 推奨)。
      - ただし ViT-bigG-14 (~1.9B params) は推論でも数十 GB 単位のメモリを消費する
        可能性があり、VRAM < 24GB 環境では OOM になる場合がある。
      - MPS では一部 reduce 演算が float32 にアップキャストされ精度が落ちることがある。
      - 安定した GPU 推論には CUDA (Colab T4/L4/A100) を推奨する。
    """
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


_PHASE0_DIR = Path(__file__).resolve().parent.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

import config as cfg
from tools.lps_utils import lps_octant, normalize_pointcloud
from tools.mesh_to_pointcloud import load_mandible_pointcloud, sample_patch_centers
from tools.render_with_marker import (
    render_mesh_with_patch_color,
    annotate_image,
    save_rendered_images,
    AZIMUTHAL_VIEWS,
)


# ── CLIP テンプレート (ULIP と同一) ───────────────────────────────────────
TEMPLATES = ["{}", "a {}", "{} part"]

# ── クローズドボキャブラリ (system_prompt.txt の閉語彙セットと同期) ───────

_BILATERAL_TERMS: list[str] = [
    # [Condylar region]
    "condylar head",
    "condylar neck",
    "pterygoid fovea",
    # [Coronoid and notch region]
    "coronoid process",
    "mandibular notch",
    "sigmoid notch",
    # [Mandibular ramus]
    "mandibular ramus lateral surface",
    "mandibular ramus medial surface",
    "mandibular foramen",
    "lingula",
    "mylohyoid groove",
    "masseteric tuberosity",
    "pterygoid tuberosity",
    # [Mandibular angle]
    "mandibular angle",
    "gonion",
    # [Retromolar and posterior body]
    "retromolar triangle",
    "external oblique ridge",
    "linea obliqua",
    "buccal shelf",
    # [Mandibular body — external]
    "mandibular body",
    "alveolar process",
    "mental foramen",
    # [Mandibular body — internal]
    "mylohyoid line",
    "linea mylohyoidea",
    "submandibular fossa",
    "sublingual fossa",
    # [Anterior / symphysis region — bilateral]
    "parasymphysis",
    "mental tubercle",
    "incisive fossa",
]

_MIDLINE_TERMS: list[str] = [
    "symphysis menti",
    "mental protuberance",
    "mental spine",
    "genial tubercle",
    "mandibular symphysis",
]

# オクタント別ボキャブラリ (ランキング対象を絞る)
# 側方オクタントはそのサイドの bilateral term のみ + 正中近接 (-A-) なら midline も追加
# 正中オクタントは両側 + midline 全語
_OCTANT_FOCUS: dict[str, list[str]] = {
    "L-P-S": ["condylar head", "condylar neck", "pterygoid fovea",
               "mandibular notch", "sigmoid notch", "coronoid process"],
    "R-P-S": ["condylar head", "condylar neck", "pterygoid fovea",
               "mandibular notch", "sigmoid notch", "coronoid process"],
    "L-A-S": ["coronoid process", "mandibular notch", "sigmoid notch",
               "alveolar process", "incisive fossa", "condylar head"],
    "R-A-S": ["coronoid process", "mandibular notch", "sigmoid notch",
               "alveolar process", "incisive fossa", "condylar head"],
    "L-P-I": ["mandibular angle", "gonion", "mandibular ramus lateral surface",
               "mandibular ramus medial surface", "mandibular foramen", "lingula",
               "mylohyoid groove", "masseteric tuberosity", "pterygoid tuberosity",
               "retromolar triangle", "external oblique ridge", "linea obliqua"],
    "R-P-I": ["mandibular angle", "gonion", "mandibular ramus lateral surface",
               "mandibular ramus medial surface", "mandibular foramen", "lingula",
               "mylohyoid groove", "masseteric tuberosity", "pterygoid tuberosity",
               "retromolar triangle", "external oblique ridge", "linea obliqua"],
    "L-A-I": ["mental foramen", "mandibular body", "alveolar process",
               "buccal shelf", "external oblique ridge", "linea obliqua",
               "mylohyoid line", "linea mylohyoidea",
               "submandibular fossa", "sublingual fossa",
               "parasymphysis", "mental tubercle"],
    "R-A-I": ["mental foramen", "mandibular body", "alveolar process",
               "buccal shelf", "external oblique ridge", "linea obliqua",
               "mylohyoid line", "linea mylohyoidea",
               "submandibular fossa", "sublingual fossa",
               "parasymphysis", "mental tubercle"],
    "M-A-S": ["alveolar process", "incisive fossa", "coronoid process"],
    "M-A-I": ["symphysis menti", "mental protuberance", "mental spine",
               "genial tubercle", "mandibular symphysis", "parasymphysis",
               "mental tubercle", "mental foramen", "mandibular body"],
    "M-P-S": ["alveolar process", "incisive fossa", "sublingual fossa"],
    "M-P-I": ["mental spine", "genial tubercle", "mylohyoid line",
               "linea mylohyoidea", "submandibular fossa", "sublingual fossa"],
}


def build_vocab_for_octant(octant: str) -> list[str]:
    """
    オクタントに対応する候補テキストリストを構築する。
    左右オクタント: "left/right X" 形式, 正中オクタント: midline 語 + "left/right X"
    """
    lr, _, _ = octant.split("-")
    focus = _OCTANT_FOCUS.get(octant, _BILATERAL_TERMS)

    vocab: list[str] = []
    if lr == "L":
        for term in focus:
            vocab.append(f"left {term}")
        # 隣接オクタントからの補完 (正中近接なら midline も追加)
        vocab += _MIDLINE_TERMS
    elif lr == "R":
        for term in focus:
            vocab.append(f"right {term}")
        vocab += _MIDLINE_TERMS
    else:  # "M"
        vocab += _MIDLINE_TERMS
        for term in focus:
            vocab.append(f"left {term}")
            vocab.append(f"right {term}")

    # 重複除去・順序保持
    seen: set[str] = set()
    result: list[str] = []
    for v in vocab:
        if v not in seen:
            seen.add(v)
            result.append(v)
    return result


# ── CLIP モデル ────────────────────────────────────────────────────────────

def load_clip_model(device: str = "cpu"):
    """
    OpenCLIP ViT-bigG-14 (ULIP と同一モデル) をロードする。

    Returns:
        model, preprocess, tokenizer
    """
    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip が必要です: pip install open-clip-torch")

    print(f"  [CLIP] Loading ViT-bigG-14 (laion2b_s39b_b160k) on {device} ...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-bigG-14", pretrained="laion2b_s39b_b160k",
    )
    tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
    model = model.to(device).eval()
    print("  [CLIP] Model loaded.")
    return model, preprocess, tokenizer


# ── 特徴量エンコード ──────────────────────────────────────────────────────

def encode_images(
    rendered_views: dict[str, np.ndarray],
    model,
    preprocess,
    device: str,
):
    """
    複数視点の画像を CLIP でエンコードし、平均プーリングで 1 つの特徴量にまとめる。

    Args:
        rendered_views: {"az000": img_np, "az045": img_np, ...}

    Returns:
        feat: (1, D) torch.Tensor  (L2 正規化済み)
    """
    import torch
    import torch.nn.functional as F
    from PIL import Image

    feats = []
    with torch.no_grad():
        for img_np in rendered_views.values():
            pil = Image.fromarray(img_np)
            tensor = preprocess(pil).unsqueeze(0).to(device)
            f = model.encode_image(tensor)           # (1, D)
            feats.append(F.normalize(f, dim=-1))

    avg = torch.stack(feats, dim=0).mean(dim=0)     # (1, D)
    return F.normalize(avg, dim=-1)


def encode_texts(
    vocab: list[str],
    model,
    tokenizer,
    device: str,
) -> "torch.Tensor":
    """
    ボキャブラリを CLIP でエンコードする (テンプレート平均プーリング)。

    Returns:
        text_feats: (V, D) torch.Tensor  (L2 正規化済み)
    """
    import torch
    import torch.nn.functional as F

    feats = []
    with torch.no_grad():
        for text in vocab:
            tokens = tokenizer([t.format(text) for t in TEMPLATES]).to(device)
            f = model.encode_text(tokens)            # (len(TEMPLATES), D)
            pooled = F.normalize(f.mean(0, keepdim=True), dim=-1)
            feats.append(pooled)

    return torch.cat(feats, dim=0)                   # (V, D)


def rank_candidates(
    img_feat,        # (1, D)
    text_feats,      # (V, D)
    vocab: list[str],
    top_k: int = 10,
) -> list[tuple[str, float]]:
    """
    画像特徴量に対してテキスト候補を cosine 類似度でランキングする。

    Returns:
        [(text, score), ...] (降順, top_k 件)
    """
    sims = (img_feat @ text_feats.T).squeeze(0)         # (V,)
    sorted_idx = sims.argsort(descending=True)[:top_k]
    return [(vocab[int(i)], float(sims[i])) for i in sorted_idx]


# ── 1 症例のパイプライン ──────────────────────────────────────────────────

def generate_clip_candidates_for_case(
    case_id: str,
    ply_path: Path,
    model,
    preprocess,
    tokenizer,
    n_sample: int,
    n_patches: int,
    patch_radius: float,
    top_k: int,
    device: str,
    save_renders: bool = True,
) -> list[dict]:
    """
    1 症例の PLY から全パッチの CLIP ランキング候補を生成する。

    Returns:
        LLM モードと同一フォーマットの候補 dict リスト
        (texts フィールドに上位 top_k 件、clip_scores フィールドに対応スコア)
    """
    print(f"\n[{case_id}] Loading: {ply_path.name}")
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    pts  = load_mandible_pointcloud(ply_path, n_sample)
    centers, _ = sample_patch_centers(pts, n_patches)

    results: list[dict] = []
    for i, center in enumerate(centers):
        octant = lps_octant(center)
        print(f"  Patch {i:03d}/{n_patches-1}  LPS: {octant}", end=" ", flush=True)

        try:
            # 8 方位レンダリング
            rendered_az = render_azimuthal_views(
                mesh, center, patch_id=i,
                patch_radius=patch_radius,
                image_size=cfg.IMAGE_SIZE,
            )

            if save_renders:
                render_out = cfg.RENDERS_DIR / "clip" / case_id / f"patch_{i:03d}"
                save_rendered_images(rendered_az, render_out)

            # CLIP ランキング
            vocab = build_vocab_for_octant(octant)
            img_feat   = encode_images(rendered_az, model, preprocess, device)
            text_feats = encode_texts(vocab, model, tokenizer, device)
            ranked     = rank_candidates(img_feat, text_feats, vocab, top_k)

            texts  = [r[0] for r in ranked]
            scores = [r[1] for r in ranked]

            print(f"→ \"{texts[0]}\" ({scores[0]:.3f})")
            results.append({
                "patch_id":         i,
                "patch_center_lps": center.tolist(),
                "lps_octant":       octant,
                "texts":            texts,
                "clip_scores":      scores,
                "source":           "clip",
            })

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append({
                "patch_id":         i,
                "patch_center_lps": center.tolist(),
                "lps_octant":       octant,
                "texts":            ["", "", "", "", ""],
                "clip_scores":      [],
                "source":           "clip",
                "error":            str(e),
            })

    return results


# ── 8 方位レンダリング ────────────────────────────────────────────────────

def render_azimuthal_views(
    mesh,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, np.ndarray]:
    """
    z 軸周り 8 方位 (0°/45°/.../315°, elev=0) のレンダリング画像を返す。
    パッチ領域は距離勾配グラデーションでハイライト。

    Returns:
        {"az000": img_np, "az045": img_np, ..., "az315": img_np}
    """
    octant = lps_octant(patch_center_lps)
    rendered: dict[str, np.ndarray] = {}
    for view_name, vp in AZIMUTHAL_VIEWS.items():
        img = render_mesh_with_patch_color(
            mesh, patch_center_lps, vp,
            patch_radius=patch_radius,
            image_size=image_size,
        )
        img = annotate_image(img, patch_id, octant, view_name)
        rendered[view_name] = img
    return rendered


# ── メイン ────────────────────────────────────────────────────────────────

def main(
    n_sample: int   = cfg.N_POINTS_FULL,
    n_patches: int  = cfg.N_PATCHES,
    patch_radius: float = cfg.PATCH_RADIUS,
    top_k: int      = cfg.CLIP_TOP_K,
    device: str     = cfg.CLIP_DEVICE,
    save_renders: bool = True,
) -> None:
    """全症例に対して CLIP ランキングを実行し candidates_clip.json に保存する"""
    model, preprocess, tokenizer = load_clip_model(device)

    ply_files = cfg.get_ply_files()
    print(f"Found {len(ply_files)} PLY files")

    # 既存の candidates_clip.json を読み込み (再開対応)
    candidates: dict[str, list] = {}
    if cfg.CANDIDATES_CLIP_JSON.exists():
        with open(cfg.CANDIDATES_CLIP_JSON, encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"Resuming: {len(candidates)} cases already processed")

    for case_id, ply_path in ply_files.items():
        if case_id in candidates:
            print(f"[SKIP] {case_id} (already processed)")
            continue

        patches = generate_clip_candidates_for_case(
            case_id, ply_path,
            model, preprocess, tokenizer,
            n_sample, n_patches, patch_radius,
            top_k, device, save_renders,
        )
        candidates[case_id] = patches

        # 随時保存 (中断してもそこまでの結果が残る)
        cfg.VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        with open(cfg.CANDIDATES_CLIP_JSON, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {cfg.CANDIDATES_CLIP_JSON}")

    print(f"\nDone. Total cases: {len(candidates)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CLIP ベースのテキスト候補ランキング")
    parser.add_argument("--n_sample",    type=int,   default=cfg.N_POINTS_FULL)
    parser.add_argument("--n_patches",   type=int,   default=cfg.N_PATCHES)
    parser.add_argument("--patch_radius",type=float, default=cfg.PATCH_RADIUS)
    parser.add_argument("--top_k",       type=int,   default=cfg.CLIP_TOP_K)
    parser.add_argument("--device",      default=cfg.CLIP_DEVICE)
    parser.add_argument("--no-renders",  action="store_true")
    args = parser.parse_args()
    main(
        n_sample=args.n_sample,
        n_patches=args.n_patches,
        patch_radius=args.patch_radius,
        top_k=args.top_k,
        device=args.device,
        save_renders=not args.no_renders,
    )
