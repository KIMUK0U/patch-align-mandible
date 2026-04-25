"""
build_anatomy_textbank.py — 採用済みラベルから OpenCLIP テキスト埋め込みバンクを構築

出力:
    data/DentalPatchData/dental_vocab/
    ├── anatomy_vocabulary.txt      # ユニークラベル一覧 (simple ラベル)
    └── anatomy_textbank.pt         # {keys: [...], emb: (K, 1280)} float32

依存: open_clip_torch, torch

実行:
    cd /path/to/ULIP_PointLLM
    python DentalPatchAligned3D/Phase0_Data/tools/build_anatomy_textbank.py \
        --device cuda
"""

import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

_PHASE0_DIR = Path(__file__).resolve().parent.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

import config as cfg

# PatchAlign3D (stage2.py) と同一テンプレート
CLIP_MODEL_NAME = "ViT-bigG-14"
CLIP_PRETRAINED = "laion2b_s39b_b160k"
TEMPLATES = ["{}", "a {}", "{} part"]


def collect_texts(entry: dict) -> list[str]:
    """
    採用済み全テキスト (accepted_texts) を返す。
    accepted_texts が存在しない旧フォーマットは mask_label にフォールバック。
    """
    accepted = entry.get("accepted_texts")
    if accepted:
        return [t for t in accepted if t]

    # 旧フォーマット互換: mask_label のみ
    ml = entry.get("mask_label", "")
    return [ml] if ml else []


def build_textbank(
    verified_path: Path,
    vocab_dir: Path,
    device: str = "cpu",
) -> None:
    with open(verified_path, encoding="utf-8") as f:
        verified: dict[str, dict] = json.load(f)

    try:
        import open_clip
    except ImportError:
        raise ImportError("open_clip_torch が必要です: pip install open_clip_torch")

    print(f"Loading CLIP model: {CLIP_MODEL_NAME} ({CLIP_PRETRAINED})")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED
    )
    tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
    clip_model = clip_model.to(device).eval()

    keys: list[str]        = []
    embs: list[torch.Tensor] = []
    seen: set[str]         = set()

    with torch.no_grad():
        for entry in verified.values():
            if entry["status"] not in ("accepted", "edited", "all_accepted"):
                continue

            for text in collect_texts(entry):
                text = text.strip()
                if not text or text in seen:
                    continue
                seen.add(text)

                # テンプレート 3 種の平均埋め込み
                prompts = [t.format(text) for t in TEMPLATES]
                tokens  = tokenizer(prompts).to(device)
                feats   = F.normalize(clip_model.encode_text(tokens), dim=-1)
                pooled  = F.normalize(feats.mean(0), dim=-1)

                keys.append(text)
                embs.append(pooled.cpu())

    if not keys:
        print("[WARNING] 採用済みラベルが 0 件です。verified_labels.json を確認してください。")
        return

    vocab_dir.mkdir(parents=True, exist_ok=True)

    # anatomy_textbank.pt
    textbank = {"keys": keys, "emb": torch.stack(embs)}
    torch.save(textbank, vocab_dir / "anatomy_textbank.pt")

    # anatomy_vocabulary.txt (simple ラベルのみ)
    simple_labels = sorted({
        entry.get("mask_label", "")
        for entry in verified.values()
        if entry["status"] in ("accepted", "edited", "all_accepted")
        and entry.get("mask_label")
    })
    (vocab_dir / "anatomy_vocabulary.txt").write_text(
        "\n".join(simple_labels), encoding="utf-8"
    )

    print(f"Textbank: {len(keys)} entries → {vocab_dir / 'anatomy_textbank.pt'}")
    print(f"Vocabulary: {len(simple_labels)} labels → {vocab_dir / 'anatomy_vocabulary.txt'}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="解剖テキストバンク構築")
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda'")
    args = parser.parse_args()

    build_textbank(
        verified_path = cfg.VERIFIED_JSON,
        vocab_dir     = cfg.VOCAB_DIR,
        device        = args.device,
    )
