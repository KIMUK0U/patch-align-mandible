"""
verify_text_labels.py — LLM 生成テキスト候補の人手確認 CLI

フロー:
  candidates.json を読み込む
    ↓
  [症例 × パッチ] を 1 件ずつ:
    ① Open3D で点群 + パッチ中心 (赤球) を表示
    ② renders/ の画像 (6 視点) をデフォルトビューアで自動表示
    ③ 端末に LLM 生成テキスト (texts[0]〜texts[4]) を表示
    ④ ユーザーが操作を選択
    ⑤ verified_labels.json に随時保存

実行:
    cd /path/to/ULIP_PointLLM
    python DentalPatchAligned3D/Phase0_Data/tools/verify_text_labels.py
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import open3d as o3d

_PHASE0_DIR = Path(__file__).resolve().parent.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

import config as cfg
from tools.lps_utils import lps_octant, normalize_pointcloud
from tools.mesh_to_pointcloud import load_mandible_pointcloud

N_TEXT_VARIANTS = 5  # texts[0] 〜 texts[4]


# ── 可視化ヘルパー ────────────────────────────────────────────────────────

def show_pointcloud_with_marker(pts_np: np.ndarray, center_lps: np.ndarray) -> None:
    """Open3D ウィンドウに点群 + 赤球マーカーを表示 (閉じると続行)"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts_np)
    pcd.paint_uniform_color([0.6, 0.6, 0.6])

    marker = o3d.geometry.TriangleMesh.create_sphere(radius=0.025)
    marker.translate(center_lps.tolist())
    marker.paint_uniform_color([1.0, 0.0, 0.0])
    marker.compute_vertex_normals()

    o3d.visualization.draw_geometries(
        [pcd, marker],
        window_name="Patch Verifier — close to continue",
        width=800, height=600,
    )


def open_renders(renders_dir: Path) -> None:
    """6 視点画像を OS デフォルトビューアで開く"""
    if not renders_dir.exists():
        print(f"  [WARNING] レンダリングが見つかりません: {renders_dir}")
        return
    for img_path in sorted(renders_dir.glob("*.png")):
        if sys.platform == "darwin":
            subprocess.Popen(["open", str(img_path)])
        elif sys.platform.startswith("linux"):
            subprocess.Popen(["xdg-open", str(img_path)])
        else:
            subprocess.Popen(["start", str(img_path)], shell=True)




# ── メインインタラクティブループ ──────────────────────────────────────────

def verify_interactive(
    candidates_path: Path,
    verified_path: Path,
    renders_base: Path,
    ply_dir: Path,
    skip_open3d: bool = False,
    skip_images: bool = False,
) -> None:
    with open(candidates_path, encoding="utf-8") as f:
        candidates: dict[str, list] = json.load(f)

    confirmed: dict[str, dict] = {}
    if verified_path.exists():
        with open(verified_path, encoding="utf-8") as f:
            confirmed = json.load(f)
        print(f"再開: {len(confirmed)} 件確認済み")

    ply_files = cfg.get_ply_files()

    bulk_accept_remaining = False  # AA コマンドで設定されると True になり残りを全採用

    for case_id, patch_list in candidates.items():
        if bulk_accept_remaining:
            break

        print(f"\n{'#'*66}")
        print(f"  症例: {case_id}  ({len(patch_list)} patches)")
        print(f"{'#'*66}")

        # 点群読み込み (可視化用)
        pts_np: np.ndarray | None = None
        if not skip_open3d and case_id in ply_files:
            try:
                pts_np = load_mandible_pointcloud(ply_files[case_id], n_sample=4096)
            except Exception as e:
                print(f"  [WARNING] 点群読み込みエラー: {e}")

        for patch_info in patch_list:
            patch_id = patch_info["patch_id"]
            key = f"{case_id}__patch{patch_id:03d}"
            if key in confirmed:
                print(f"  [SKIP] {key}")
                continue

            octant = patch_info.get("lps_octant", "?-?-?")
            center = np.array(patch_info["patch_center_lps"], dtype=np.float32)

            # エラーパッチはスキップ確認
            texts = patch_info.get("texts", [])
            if patch_info.get("error") or not texts:
                print(f"\n  [Patch #{patch_id:03d}] エラーまたは空ラベル — スキップしますか? [y/n]")
                if input("  → ").strip().lower() != "n":
                    confirmed[key] = {"case_id": case_id, "patch_id": patch_id, "status": "excluded", "reason": "error/empty"}
                    _save(confirmed, verified_path)
                    continue

            print(f"\n{'='*66}")
            print(f"  症例: {case_id}  |  Patch #{patch_id:03d}  |  LPS: {octant}")
            print(f"{'='*66}")

            # Open3D 可視化
            if pts_np is not None and not skip_open3d:
                print("  [Open3D] 点群 + パッチマーカーを表示 → 閉じると続行")
                show_pointcloud_with_marker(pts_np, center)

            # レンダリング画像を自動表示
            renders_dir = renders_base / case_id / f"patch_{patch_id:03d}"
            if not skip_images:
                print("  [画像] レンダリング 6 視点を表示中...")
                open_renders(renders_dir)

            # テキスト候補を表示・編集ループ
            working_texts = list(texts)   # 編集用コピー

            if bulk_accept_remaining:
                # AA コマンド発動中: 残りを全採用してスキップ
                confirmed[key] = _make_entry(case_id, patch_id, octant,
                                              working_texts, texts, "all_accepted")
                _save(confirmed, verified_path)
                continue

            while True:
                print()
                for i, text in enumerate(working_texts, 1):
                    print(f"  [{i}] \"{text}\"")
                print()
                print(f"  □ 解剖学的名称は正しいか?")
                print(f"  □ 左右 ({octant} → {'Left' if octant.startswith('L') else 'Right'} 側) は正しいか?")
                print(f"  □ Hallucination がないか?")
                print()
                n = len(working_texts)
                print(f"  操作: [a]=全採用  [数字の組合せ]=部分採用 (例:135)  [eN]=N番を編集 (例:e2)")
                print(f"        [r]=再生成メモ  [x]=除外  [AA]=残り全パッチを全採用してスキップ")

                choice = input("  → ").strip()

                if choice.lower() == "aa":
                    # 残件数を計算
                    pending = sum(
                        1 for cid2, plist2 in candidates.items()
                        for p2 in plist2
                        if f"{cid2}__patch{p2['patch_id']:03d}" not in confirmed
                    )
                    print(f"  ⚠  残り {pending} 件を全て全採用します。よろしいですか? [y/N]")
                    if input("  → ").strip().lower() == "y":
                        # 現在のパッチも全採用して一括モードへ
                        confirmed[key] = _make_entry(case_id, patch_id, octant,
                                                      working_texts, texts, "all_accepted")
                        _save(confirmed, verified_path)
                        bulk_accept_remaining = True
                        print("  ✓ 一括採用モード開始。残りを全採用します。")
                    break

                elif choice.lower() == "a":
                    confirmed[key] = _make_entry(case_id, patch_id, octant,
                                                  working_texts, texts, "all_accepted")
                    print(f"  ✓ 全採用 ({len(working_texts)} テキスト)")
                    break

                elif choice.lower().startswith("e") and len(choice) >= 2 and choice[1:].isdigit():
                    idx = int(choice[1:]) - 1
                    if 0 <= idx < len(working_texts):
                        print(f"  現在: \"{working_texts[idx]}\"")
                        edited = input(f"  [{idx+1}] 編集後テキストを入力: ").strip()
                        if not edited:
                            print("  ⚠  空のテキストは入力できません")
                            continue
                        if len(edited) > 200:
                            print(f"  ⚠  {len(edited)} chars — 長いです。続けますか? [y/n]")
                            if input("  → ").strip().lower() != "y":
                                continue
                        working_texts[idx] = edited
                        print(f"  → texts[{idx}] を更新しました")
                    else:
                        print(f"  ⚠  無効な番号です (1〜{n})")

                elif choice and all(c.isdigit() for c in choice):
                    indices = sorted({int(c) - 1 for c in choice
                                      if 1 <= int(c) <= len(working_texts)})
                    if not indices:
                        print(f"  ⚠  有効な番号がありません (1〜{n})")
                        continue
                    accepted = [working_texts[i] for i in indices]
                    confirmed[key] = _make_entry(case_id, patch_id, octant,
                                                  accepted, texts, "accepted")
                    nums = "".join(str(i+1) for i in indices)
                    print(f"  ✓ 部分採用 [{nums}]: {len(accepted)} テキスト")
                    break

                elif choice.lower() == "r":
                    note = input("  再生成メモ (問題点): ").strip()
                    confirmed[key] = {
                        "case_id":    case_id, "patch_id": patch_id,
                        "lps_octant": octant,
                        "mask_label": working_texts[0] if working_texts else "",
                        "accepted_texts": [],
                        "all_candidates": {"texts": texts},
                        "status": "needs_regeneration",
                        "regeneration_note": note,
                    }
                    print("  → 再生成待ちとしてマーク")
                    break

                elif choice.lower() == "x":
                    confirmed[key] = {"case_id": case_id, "patch_id": patch_id, "status": "excluded"}
                    print("  ✗ 除外")
                    break

                else:
                    print(f"  無効な入力です。[a / 数字の組合せ / eN / r / x / AA] を入力してください")

            _save(confirmed, verified_path)
            if bulk_accept_remaining:
                break  # 内側ループ (patch_list) を抜ける

    print(f"\n完了。confirmed: {len(confirmed)} 件 → {verified_path}")


def _make_entry(case_id, patch_id, octant, accepted_texts, original_texts, status):
    """
    accepted_texts  : 採用するテキストのリスト (編集済みを含む)
    original_texts  : LLM 生成の元テキストリスト (all_candidates 保存用)
    """
    return {
        "case_id":        case_id,
        "patch_id":       patch_id,
        "lps_octant":     octant,
        "mask_label":     accepted_texts[0] if accepted_texts else "",
        "accepted_texts": accepted_texts,
        "all_candidates": {"texts": original_texts},
        "status":         status,
    }


def _save(confirmed: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(confirmed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LLM テキスト候補の人手確認")
    parser.add_argument("--no-open3d", action="store_true", help="Open3D 可視化をスキップ")
    parser.add_argument("--no-images", action="store_true", help="画像表示をスキップ")
    args = parser.parse_args()

    verify_interactive(
        candidates_path = cfg.CANDIDATES_JSON,
        verified_path   = cfg.VERIFIED_JSON,
        renders_base    = cfg.RENDERS_DIR,
        ply_dir         = cfg.PLY_DIR,
        skip_open3d     = args.no_open3d,
        skip_images     = args.no_images,
    )
