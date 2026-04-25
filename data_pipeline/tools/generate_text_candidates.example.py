"""
generate_text_candidates.example.py — LLM API テキスト候補自動生成 (設定テンプレート)

このファイルをコピーして generate_text_candidates.py を作成し、
API キーを設定してから実行してください。
generate_text_candidates.py は .gitignore で除外されています。

    cp tools/generate_text_candidates.example.py tools/generate_text_candidates.py
    # generate_text_candidates.py の OPENAI_API_KEY を設定する

実行:
    cd /path/to/ULIP_PointLLM
    python DentalPatchAligned3D/Phase0_Data/tools/generate_text_candidates.py \
        --n_patches 32 \
        --n_sample 8192

出力:
    data/DentalPatchData/dental_vocab/candidates.json
    data/DentalPatchData/renders/<case_id>/patch_<NNN>/*.png
"""

import sys
import json
import traceback
from pathlib import Path

import numpy as np
import open3d as o3d

# ── API キー設定 (ここを編集してください) ─────────────────────────────────
OPENAI_API_KEY = "YOUR_OPENAI_API_KEY_HERE"
OPENAI_MODEL   = "gpt-5.4-2026-03-05"
# ── 代替: Gemini を使う場合 ───────────────────────────────────────────────
# GEMINI_API_KEY = "YOUR_GEMINI_API_KEY_HERE"
# GEMINI_MODEL   = "gemini-1.5-pro"

# ── プロジェクトルートを sys.path に追加 ─────────────────────────────────
_PHASE0_DIR = Path(__file__).resolve().parent.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

import config as cfg
from tools.lps_utils import (
    lps_octant,
    octant_to_lr,
    get_normalization_params,
    apply_normalization,
)
from tools.mesh_to_pointcloud import (
    load_mandible_pointcloud,
    sample_patch_centers,
)
from tools.render_with_marker import (
    render_all_views,
    render_all_views_stl,
    select_views_for_llm,
    image_to_base64,
    save_rendered_images,
)


# ── OpenAI クライアント初期化 ─────────────────────────────────────────────
def build_openai_client():
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("openai パッケージが必要です: pip install openai")
    return OpenAI(api_key=OPENAI_API_KEY)


# ── プロンプト構築 ────────────────────────────────────────────────────────
def build_user_message(
    rendered: dict[str, np.ndarray],
    patch_id: int,
    center_lps: np.ndarray,
    prompt_template: dict,
    max_views: int = 10,
) -> list[dict]:
    """GPT-4o 向けマルチモーダルメッセージを構築"""
    octant         = lps_octant(center_lps)
    lr_side, opp_side = octant_to_lr(octant)

    pos_desc = prompt_template["position_descriptions"].get(
        octant, f"unknown region ({octant})"
    )

    if octant.startswith("M"):
        sym_note = prompt_template["midline_symmetry_note"]
    else:
        sym_note = prompt_template["symmetry_note_template"].format(
            lr_side=lr_side, opposite_side=opp_side
        )

    user_text = prompt_template["user_template"].format(
        lps_octant=octant,
        position_description=pos_desc,
        symmetry_note=sym_note,
    )

    user_text += (
        "\n\nNote: No reference diagram is provided. Rely on your anatomical knowledge "
        "and the 3D rendered views to identify the specific feature. Pay close attention "
        "to whether the region is a notch, a process, a neck, or a fossa."
    )

    content: list[dict] = []

    _API_VIEWS = ["anterior", "az225", "az315", "posterior"]
    selected_views = [v for v in _API_VIEWS if v in rendered]
    for view_name in selected_views:
        img_b64 = image_to_base64(rendered[view_name])
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,{img_b64}"},
        })
        content.append({"type": "text", "text": f"[View: {view_name}]"})

    content.append({"type": "text", "text": user_text})
    return content


def call_gpt4o(system_prompt: str, user_content: list, client) -> dict:
    """GPT-4o Vision API を呼び出し JSON (text_1..text_5) を取得"""
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_content},
        ],
        max_completion_tokens=600,
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content
    return json.loads(raw)


# ── メインパイプライン ────────────────────────────────────────────────────
def generate_candidates_for_case(
    case_id: str,
    ply_path: Path,
    system_prompt: str,
    prompt_template: dict,
    client,
    n_sample: int,
    n_patches: int,
    mode: str = "pointcloud",
    stl_path: Path | None = None,
) -> list[dict]:
    """
    1 症例の PLY (または STL) から全パッチの候補テキストを生成する。

    mode="pointcloud": PLY メッシュを散布点でレンダリング (既存の動作)
    mode="stl":        PLY からパッチ中心を算出し、STL メッシュを三角形面でレンダリング

    Returns: patch ごとの候補 dict のリスト
    """
    print(f"\n[{case_id}] Loading: {ply_path.name}  mode={mode}")

    if mode == "stl":
        # PLY は既に normalize 済みなので load_mandible_pointcloud でそのまま取得する。
        # STL は絶対座標 (mm単位) のため独立して正規化する。
        # 同じ解剖構造なので両者の正規化後座標空間はほぼ一致する。
        pts = load_mandible_pointcloud(ply_path, n_sample)

        render_mesh   = o3d.io.read_triangle_mesh(str(stl_path))
        stl_verts_raw = np.asarray(render_mesh.vertices, dtype=np.float64)
        stl_norm_mean, stl_norm_scale = get_normalization_params(stl_verts_raw)
        stl_verts_norm = apply_normalization(stl_verts_raw, stl_norm_mean, stl_norm_scale)
        print(f"  STL: {stl_path.name}  faces={len(render_mesh.triangles):,}")
    else:
        render_mesh    = o3d.io.read_triangle_mesh(str(ply_path))
        pts            = load_mandible_pointcloud(ply_path, n_sample)
        stl_verts_norm = None

    centers, _ = sample_patch_centers(pts, n_patches)

    results = []
    for i, center in enumerate(centers):
        octant = lps_octant(center)
        print(f"  Patch {i:03d}/{n_patches-1}  LPS: {octant}", end=" ", flush=True)

        try:
            if mode == "stl":
                rendered = render_all_views_stl(
                    render_mesh, stl_verts_norm, center,
                    patch_id=i, patch_radius=cfg.PATCH_RADIUS,
                )
            else:
                rendered = render_all_views(
                    render_mesh, center, patch_id=i, patch_radius=cfg.PATCH_RADIUS,
                )

            # レンダリング画像を保存 (stl モードのみサブディレクトリを追加)
            render_out = (
                cfg.RENDERS_DIR / "stl" / case_id / f"patch_{i:03d}"
                if mode == "stl"
                else cfg.RENDERS_DIR / case_id / f"patch_{i:03d}"
            )
            save_rendered_images(rendered, render_out)

            user_content = build_user_message(rendered, i, center, prompt_template)
            label_dict   = call_gpt4o(system_prompt, user_content, client)

            texts = [label_dict.get(f"text_{k}", "") for k in range(1, 6)]

            entry = {
                "patch_id":         i,
                "patch_center_lps": center.tolist(),
                "lps_octant":       octant,
                "texts":            texts,
            }
            results.append(entry)
            print(f"→ \"{texts[0]}\"")

        except Exception as e:
            print(f"  [ERROR] {e}")
            traceback.print_exc()
            results.append({
                "patch_id":         i,
                "patch_center_lps": center.tolist(),
                "lps_octant":       octant,
                "texts":            ["", "", "", "", ""],
                "error":            str(e),
            })

    return results


def main(n_sample: int = 8192, n_patches: int = 32, mode: str = "pointcloud"):
    # クライアント初期化
    client = build_openai_client()

    # プロンプト読み込み
    with open(cfg.SYSTEM_PROMPT_PATH, encoding="utf-8") as f:
        system_prompt = f.read()
    with open(cfg.USER_PROMPT_TMPL_PATH, encoding="utf-8") as f:
        prompt_template = json.load(f)

    # ファイル一覧
    ply_files = cfg.get_ply_files()
    print(f"Found {len(ply_files)} PLY files")

    stl_files: dict[str, Path] = {}
    if mode == "stl":
        stl_files = cfg.get_stl_files()
        print(f"Found {len(stl_files)} STL files")

    # 既存の candidates.json を読み込み (再開対応)
    candidates: dict[str, list] = {}
    if cfg.CANDIDATES_JSON.exists():
        with open(cfg.CANDIDATES_JSON, encoding="utf-8") as f:
            candidates = json.load(f)
        print(f"Resuming: {len(candidates)} cases already processed")

    for case_id, ply_path in ply_files.items():
        if case_id in candidates:
            print(f"[SKIP] {case_id} (already processed)")
            continue

        stl_path = stl_files.get(case_id) if mode == "stl" else None
        if mode == "stl" and stl_path is None:
            print(f"[SKIP] {case_id} — STL not found")
            continue

        patches = generate_candidates_for_case(
            case_id, ply_path, system_prompt, prompt_template,
            client, n_sample, n_patches,
            mode=mode, stl_path=stl_path,
        )
        candidates[case_id] = patches

        # 随時保存 (中断してもそこまでの結果が残る)
        cfg.VOCAB_DIR.mkdir(parents=True, exist_ok=True)
        with open(cfg.CANDIDATES_JSON, "w", encoding="utf-8") as f:
            json.dump(candidates, f, ensure_ascii=False, indent=2)
        print(f"  Saved → {cfg.CANDIDATES_JSON}")

    print(f"\nDone. Total cases: {len(candidates)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_sample",  type=int, default=8192,
                        help="メッシュからサンプリングする点数")
    parser.add_argument("--n_patches", type=int, default=32,
                        help="1 症例あたりのパッチ数")
    parser.add_argument("--mode", choices=["pointcloud", "stl"], default="pointcloud",
                        help="pointcloud: PLY 点群散布図レンダリング (デフォルト) / "
                             "stl: STL メッシュ三角形面レンダリング")
    args = parser.parse_args()
    main(args.n_sample, args.n_patches, args.mode)
