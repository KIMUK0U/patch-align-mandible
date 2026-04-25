"""
run_phase0.py — Phase 0 データセット作成パイプライン ランナー

3 工程を順に実行するか、個別に実行するかを --step で指定する。

【--mode llm (デフォルト)】
  工程 1: LLM API (GPT-4o 等) によるテキスト候補自動生成 → candidates.json
  工程 2: 人手確認 (インタラクティブ CLI) → verified_labels.json
  工程 3: PatchAlign3D 互換フォーマット変換 + テキストバンク構築

【--mode clip】
  工程 1: OpenCLIP ViT-bigG-14 による z 軸周り 8 方位レンダリング + ランキング
          → candidates_clip.json  (API キー不要)
  工程 2: 人手確認 (candidates_clip.json を使用) → verified_labels_clip.json
  工程 3: 同上 (verified_labels_clip.json を使用)

【--render_mode stl】 (--mode llm と組み合わせて使用)
  工程 1 のレンダリングを PLY 点群散布図ではなく STL メッシュ三角形面で行う。
  パッチ中心算出には引き続き PLY 点群を使用する。
  レンダリング画像の保存先: renders/stl/<case_id>/patch_NNN/

使い方:
    # LLM モード (デフォルト、点群レンダリング)
    python run_phase0.py --step all
    python run_phase0.py --step 1 --n_patches 32
    python run_phase0.py --step 2
    python run_phase0.py --step 3 --device cuda

    # LLM モード + STL メッシュレンダリング
    python run_phase0.py --render_mode stl --step 1
    python run_phase0.py --render_mode stl --step 2

    # CLIP モード (API キー不要)
    python run_phase0.py --mode clip --step all
    python run_phase0.py --mode clip --step 1 --device cuda --top_k 10
    python run_phase0.py --mode clip --step 2
    python run_phase0.py --mode clip --step 3

    # テスト: 点群読み込み + レンダリングのみ確認 (API 不要)
    python run_phase0.py --step test
"""

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config as cfg


# ── 各工程 ────────────────────────────────────────────────────────────────

def step1_generate(
    n_sample: int,
    n_patches: int,
    render_mode: str = "pointcloud",
) -> None:
    """工程 1 (LLM モード): GPT-4o API によるテキスト候補自動生成"""
    gen_script = _HERE / "tools" / "generate_text_candidates.py"
    if not gen_script.exists():
        print("[ERROR] tools/generate_text_candidates.py が見つかりません。")
        print("  generate_text_candidates.example.py をコピーして API キーを設定してください:")
        print("  cp tools/generate_text_candidates.example.py tools/generate_text_candidates.py")
        sys.exit(1)

    # 動的インポート (API キー入りファイルを直接 import)
    import importlib.util
    spec = importlib.util.spec_from_file_location("gen", gen_script)
    mod  = importlib.util.module_from_spec(spec)   # type: ignore
    spec.loader.exec_module(mod)                    # type: ignore
    mod.main(n_sample=n_sample, n_patches=n_patches, mode=render_mode)


def step1_clip(
    n_sample: int,
    n_patches: int,
    patch_radius: float,
    top_k: int,
    device: str,
) -> None:
    """工程 1 (CLIP モード): OpenCLIP ViT-bigG-14 による 8 方位レンダリング + ランキング"""
    from tools.clip_text_ranker import main as clip_main
    clip_main(
        n_sample=n_sample,
        n_patches=n_patches,
        patch_radius=patch_radius,
        top_k=top_k,
        device=device,
    )


def step2_verify(
    skip_open3d: bool,
    skip_images: bool,
    mode: str = "llm",
    render_mode: str = "pointcloud",
) -> None:
    """工程 2: 人手確認 (mode="llm" → candidates.json, mode="clip" → candidates_clip.json)"""
    from tools.verify_text_labels import verify_interactive
    candidates_path = cfg.CANDIDATES_CLIP_JSON if mode == "clip" else cfg.CANDIDATES_JSON
    verified_path   = cfg.VERIFIED_CLIP_JSON   if mode == "clip" else cfg.VERIFIED_JSON
    if mode == "clip":
        renders_base = cfg.RENDERS_DIR / "clip"
    elif render_mode == "stl":
        renders_base = cfg.RENDERS_DIR / "stl"
    else:
        renders_base = cfg.RENDERS_DIR

    if not candidates_path.exists():
        print(f"[ERROR] 候補ファイルが見つかりません: {candidates_path}")
        print(f"  先に工程 1 (--mode {mode} --step 1) を実行してください。")
        sys.exit(1)

    verify_interactive(
        candidates_path = candidates_path,
        verified_path   = verified_path,
        renders_base    = renders_base,
        ply_dir         = cfg.PLY_DIR,
        skip_open3d     = skip_open3d,
        skip_images     = skip_images,
    )


def step3_convert(
    n_final: int,
    patch_radius: float,
    n_sample: int,
    val_ratio: float,
    device: str,
    mode: str = "llm",
) -> None:
    """工程 3: PatchAlign3D 互換フォーマット変換 + テキストバンク構築"""
    from tools.build_patchalign_dataset import build_dataset
    from tools.build_anatomy_textbank  import build_textbank

    verified_path = cfg.VERIFIED_CLIP_JSON if mode == "clip" else cfg.VERIFIED_JSON
    if not verified_path.exists():
        print(f"[ERROR] verified_labels が見つかりません: {verified_path}")
        print(f"  先に工程 2 (--mode {mode} --step 2) を完了してください。")
        sys.exit(1)

    print("=== 工程 3-A: データセット変換 ===")
    build_dataset(
        verified_path  = verified_path,
        ply_dir        = cfg.PLY_DIR,
        dataset_root   = cfg.DATASET_ROOT,
        split_dir      = cfg.SPLIT_DIR,
        n_final        = n_final,
        patch_radius   = patch_radius,
        n_sample       = n_sample,
        val_ratio      = val_ratio,
    )

    print("\n=== 工程 3-B: テキストバンク構築 ===")
    build_textbank(
        verified_path = verified_path,
        vocab_dir     = cfg.VOCAB_DIR,
        device        = device,
    )


def step_test() -> None:
    """テスト: API 不要で PLY 読み込み + レンダリング確認"""
    import open3d as o3d
    from tools.mesh_to_pointcloud import load_mandible_pointcloud, sample_patch_centers
    from tools.render_with_marker import render_all_views, save_rendered_images
    from tools.lps_utils import lps_octant

    ply_files = cfg.get_ply_files()
    if not ply_files:
        print(f"[ERROR] PLY が見つかりません: {cfg.PLY_DIR}")
        return

    # 最初の 1 ファイルでテスト
    case_id, ply_path = next(iter(ply_files.items()))
    print(f"テスト対象: {case_id}  ({ply_path})")

    pts = load_mandible_pointcloud(ply_path, n_sample=4096)
    print(f"  点群: {pts.shape},  range=[{pts.min():.3f}, {pts.max():.3f}]")

    centers, _ = sample_patch_centers(pts, n_patches=4)
    print(f"  パッチ中心: {centers.shape}")
    for i, c in enumerate(centers):
        print(f"    Patch {i}: {c}  LPS={lps_octant(c)}")

    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    print("  レンダリング中 (Patch 0) ...")
    rendered = render_all_views(mesh, centers[0], patch_id=0)
    out_dir  = cfg.RENDERS_DIR / "test" / "patch_000"
    save_rendered_images(rendered, out_dir)
    print(f"  保存完了: {out_dir}")
    print("テスト完了。")


# ── エントリポイント ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Phase 0: 歯科解剖データセット作成パイプライン",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--mode",  choices=["llm", "clip"], default="llm",
                        help="テキスト生成モード: llm=GPT-4o API, clip=OpenCLIP ランキング (default: llm)")
    parser.add_argument("--render_mode", choices=["pointcloud", "stl"], default="pointcloud",
                        help="[llm --step 1/2] レンダリングモード: "
                             "pointcloud=PLY 点群散布図 (デフォルト), stl=STL メッシュ三角形面")
    parser.add_argument("--step",  choices=["1", "2", "3", "all", "test"],
                        default="all",
                        help="実行する工程 (default: all)")
    parser.add_argument("--n_sample",     type=int,   default=cfg.N_POINTS_FULL,
                        help="初期点群サンプリング数")
    parser.add_argument("--n_patches",    type=int,   default=cfg.N_PATCHES,
                        help="1 症例あたりのパッチ数")
    parser.add_argument("--n_final",      type=int,   default=cfg.N_POINTS_FINAL,
                        help="FPS 後の最終点数")
    parser.add_argument("--patch_radius", type=float, default=cfg.PATCH_RADIUS,
                        help="パッチマスク半径 (正規化座標)")
    parser.add_argument("--val_ratio",    type=float, default=0.2,
                        help="検証データの割合")
    parser.add_argument("--device",       default="cpu",
                        help="デバイス (cpu/cuda)。CLIP モードの CLIP エンコード + テキストバンク構築に使用")
    parser.add_argument("--top_k",        type=int,   default=cfg.CLIP_TOP_K,
                        help="[CLIP モード] 提示する上位候補数 (default: %(default)s)")
    parser.add_argument("--no-open3d",    action="store_true",
                        help="[工程 2] Open3D 可視化をスキップ")
    parser.add_argument("--no-images",    action="store_true",
                        help="[工程 2] 画像表示をスキップ")
    args = parser.parse_args()

    print(f"モード           : {args.mode.upper()}")
    print(f"レンダリング     : {args.render_mode}")
    print(f"PLY ディレクトリ : {cfg.PLY_DIR}")
    print(f"出力ルート       : {cfg.DATA_ROOT}")
    print()

    if args.step in ("1", "all"):
        print("=" * 60)
        if args.mode == "clip":
            print("工程 1 [CLIP]: OpenCLIP 8 方位レンダリング + ランキング")
        else:
            print("工程 1 [LLM]: GPT-4o API テキスト候補自動生成")
        print("=" * 60)
        if args.mode == "clip":
            step1_clip(
                args.n_sample, args.n_patches,
                args.patch_radius, args.top_k, args.device,
            )
        else:
            step1_generate(args.n_sample, args.n_patches, args.render_mode)

    if args.step in ("2", "all"):
        print("\n" + "=" * 60)
        print("工程 2: 人手確認")
        print("=" * 60)
        step2_verify(args.no_open3d, args.no_images, mode=args.mode, render_mode=args.render_mode)

    if args.step in ("3", "all"):
        print("\n" + "=" * 60)
        print("工程 3: フォーマット変換 + テキストバンク構築")
        print("=" * 60)
        step3_convert(
            args.n_final, args.patch_radius,
            args.n_sample, args.val_ratio, args.device,
            mode=args.mode,
        )

    if args.step == "test":
        print("=" * 60)
        print("テスト: 点群読み込み + レンダリング")
        print("=" * 60)
        step_test()

    print("\nPhase 0 完了。")


if __name__ == "__main__":
    main()
