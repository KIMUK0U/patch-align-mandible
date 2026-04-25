"""
config.py — Phase 0 パス設定 (ローカル / Google Colab 自動切り替え)

ローカル実行:   python tools/generate_text_candidates.py
Colab 実行:    Google Drive をマウント後に同じスクリプトを実行
"""

import os
from pathlib import Path

# ── このファイルの場所から相対的にパスを解決 ─────────────────────────────
# config.py: .../ULIP_PointLLM/DentalPatchAligned3D/Phase0_Data/config.py
_HERE = Path(__file__).resolve().parent  # .../Phase0_Data/

# ── 実行環境の検出 ────────────────────────────────────────────────────────
_ON_COLAB = os.path.exists("/content")

# ── プロジェクトルート (ULIP_PointLLM) ───────────────────────────────────
if _ON_COLAB:
    # Google Colab: Drive をマウントしてから実行
    # !from google.colab import drive; drive.mount('/content/drive')
    _DRIVE_ROOT = Path("/content/drive/MyDrive")
    PROJECT_ROOT = _DRIVE_ROOT / "2026研究/ULIP_PointLLM"
    PLY_DIR = _DRIVE_ROOT / "2026研究/Dataset/6167726/STLs/STLs/normPLYs"
    STL_DIR = _DRIVE_ROOT / "2026研究/Dataset/6167726/STLs/STLs"
else:
    # ローカル: __file__ から相対パスで解決
    PROJECT_ROOT = _HERE.parent.parent          # .../ULIP_PointLLM/
    PLY_DIR = PROJECT_ROOT.parent / "Dataset/6167726/STLs/STLs/normPLYs"
    STL_DIR = PROJECT_ROOT.parent / "Dataset/6167726/STLs/STLs"

# ── Phase 0 スクリプトディレクトリ ──────────────────────────────────────
PHASE0_DIR = _HERE

# ── プロンプトファイル ────────────────────────────────────────────────────
PROMPTS_DIR = PHASE0_DIR / "prompts"
SYSTEM_PROMPT_PATH    = PROMPTS_DIR / "system_prompt.txt"
USER_PROMPT_TMPL_PATH = PROMPTS_DIR / "user_prompt_template.json"

# ── 出力データルート ─────────────────────────────────────────────────────
DATA_ROOT    = PROJECT_ROOT / "data/DentalPatchData"
RENDERS_DIR  = DATA_ROOT / "renders"

DATASET_ROOT = DATA_ROOT / "dental_dataset/labeled"
POINTS_DIR   = DATASET_ROOT / "points"
RENDERED_DIR = DATASET_ROOT / "rendered"
SPLIT_DIR    = DATASET_ROOT / "split"

VOCAB_DIR    = DATA_ROOT / "dental_vocab"
CANDIDATES_JSON      = VOCAB_DIR / "candidates.json"
CANDIDATES_CLIP_JSON = VOCAB_DIR / "candidates_clip.json"
VERIFIED_JSON        = VOCAB_DIR / "verified_labels.json"
VERIFIED_CLIP_JSON   = VOCAB_DIR / "verified_labels_clip.json"
VOCAB_TXT            = VOCAB_DIR / "anatomy_vocabulary.txt"
TEXTBANK_PT          = VOCAB_DIR / "anatomy_textbank.pt"

# ── PLY ファイル一覧 (10 症例 × 2 側面 = 20 ファイル) ─────────────────
def get_ply_files() -> dict[str, Path]:
    """
    Returns: {"Pat 1a_norm": Path(...), "Pat 1b_norm": Path(...), ...}
    """
    files = {}
    for p in sorted(PLY_DIR.glob("*.ply")):
        key = p.stem.replace("_sample", "")
        files[key] = p
    return files


def get_stl_files() -> dict[str, Path]:
    """
    PLY の case_id をキーにした STL パスのマッピング。
    PLY stem の "_norm" サフィックスを除いた名前で STL を検索する。
    例: "Pat 1a_norm" → STL_DIR / "Pat 1a.stl"

    Returns: {"Pat 1a_norm": Path(...), ...}  (STL が存在するもののみ)
    """
    files = {}
    for ply_case_id in get_ply_files():
        stl_stem = ply_case_id.replace("_norm", "")
        stl_path = STL_DIR / f"{stl_stem}.stl"
        if stl_path.exists():
            files[ply_case_id] = stl_path
    return files


# ── パラメータ ────────────────────────────────────────────────────────────
N_POINTS_FULL  = 2048   # 初期サンプリング点数 (メッシュから)
N_POINTS_FINAL = 2048   # FPS 後の最終点数 (PatchAlign3D 要求)
N_PATCHES      = 32     # 1 症例あたりの LLM ラベリング対象パッチ数
MIN_PER_PATCH  = 32     # 各パッチに割り当てる最低点数 (mask_aware_fps 用)
PATCH_RADIUS   = 0.15   # パッチ中心からのマスク半径 (正規化座標)
IMAGE_SIZE     = (512, 512)

# ── CLIP モード設定 ───────────────────────────────────────────────────────
CLIP_TOP_K  = 10     # ランキングで提示する上位候補数 (verify で選択)
# デバイスは実行時に clip_text_ranker.auto_device() で自動検出する。
# 明示的に指定したい場合は --device cuda / --device mps / --device cpu を使う。
CLIP_DEVICE = "cpu"  # フォールバックデフォルト
