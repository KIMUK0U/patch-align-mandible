#!/usr/bin/env python3
"""
label_patches.py — 下顎骨パッチ手動解剖構造ラベリングツール

使い方:
    # PLY ファイルを直接指定
    python label_patches.py --ply <path/to/file.ply>

    # config から症例を選択 (Pat 1a, Pat 2b, ...)
    python label_patches.py --case "Pat 1a"

    # パラメータ変更
    python label_patches.py --ply <file.ply> --n_sample 8192 --n_fps 64 --n_neighbors 128

    # 利用可能な症例を一覧表示
    python label_patches.py --list

操作フロー:
    可視化ウィンドウで Q を押す → コマンド入力 → (必要なら vis で再可視化)

コマンド一覧:
    vis              再可視化 (現在の選択パッチをオレンジで表示)
    label <名前>     現在の選択パッチにラベルを付ける
                     複数名称: label 下顎頭; 関節頭; condylar head
    add              現在の選択に最も近い未ラベルパッチを1つ追加
    remove           現在の選択から最も遠いパッチを1つ削除
    skip             現在の選択パッチをスキップ (ラベルなし)
    next             選択を変えずに次の未ラベルパッチへ移動
    list             全パッチの状態一覧
    info             現在の選択の詳細情報
    save             ラベルを保存
    quit             保存して終了
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

# ── パス設定 ──────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_PHASE0_DIR = _HERE.parent
if str(_PHASE0_DIR) not in sys.path:
    sys.path.insert(0, str(_PHASE0_DIR))

from tools.lps_utils import normalize_pointcloud, fps_numpy


# ── 定数 ──────────────────────────────────────────────────────────────────
COLOR_BG    = np.array([0.0, 0.0, 0.0])   # 背景: 黒
COLOR_BODY  = np.array([1.0, 1.0, 1.0])   # 下顎骨: 白
COLOR_PATCH = np.array([1.0, 0.55, 0.0])  # 選択パッチ: オレンジ
POINT_SIZE  = 2.5


# ── 点群読み込み ──────────────────────────────────────────────────────────

def load_pointcloud(ply_path: Path, n_sample: int) -> np.ndarray:
    """PLY メッシュ → 正規化済み点群 (n_sample, 3) float32"""
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    if not mesh.has_vertices():
        raise ValueError(f"Empty mesh: {ply_path}")
    mesh.compute_vertex_normals()

    if len(mesh.triangles) > 0:
        pcd = mesh.sample_points_uniformly(number_of_points=n_sample)
        pts = np.asarray(pcd.points, dtype=np.float32)
    else:
        pts = np.asarray(mesh.vertices, dtype=np.float32)
        if len(pts) > n_sample:
            rng = np.random.default_rng(42)
            idx = rng.choice(len(pts), n_sample, replace=False)
            pts = pts[idx]

    return normalize_pointcloud(pts)


# ── FPS + 近傍 ────────────────────────────────────────────────────────────

def compute_fps_centers(pts: np.ndarray, n_fps: int) -> tuple[np.ndarray, np.ndarray]:
    """FPS で n_fps 個の代表点を選ぶ。Returns (centers (n_fps,3), indices (n_fps,))"""
    rng = np.random.default_rng(42)
    idx = fps_numpy(pts, n_fps, rng)
    return pts[idx], idx


def find_k_nearest_indices(pts: np.ndarray, center: np.ndarray, k: int) -> np.ndarray:
    """center に最も近い k 点のインデックスを返す"""
    k = min(k, len(pts))
    dists = np.sum((pts - center) ** 2, axis=1)
    return np.argpartition(dists, k - 1)[:k]


def gather_patch_point_indices(
    pts: np.ndarray,
    centers: np.ndarray,
    patch_ids: list[int],
    n_neighbors: int,
) -> np.ndarray:
    """指定パッチ群の近傍点インデックスをまとめて返す (重複なし)"""
    selected: set[int] = set()
    for pid in patch_ids:
        nbrs = find_k_nearest_indices(pts, centers[pid], n_neighbors)
        selected.update(nbrs.tolist())
    return np.array(sorted(selected), dtype=np.int64)


# ── 可視化 ────────────────────────────────────────────────────────────────

def visualize(pts: np.ndarray, highlight_indices: np.ndarray, title: str = "") -> None:
    """
    点群を Open3D で表示。
    - 全点: 白 (下顎骨)
    - highlight_indices の点: オレンジ (選択パッチ + 近傍点)
    - 背景: 黒
    Q キーで閉じる
    """
    colors = np.tile(COLOR_BODY, (len(pts), 1)).astype(np.float32)
    if len(highlight_indices) > 0:
        colors[highlight_indices] = COLOR_PATCH

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    win_title = f"Patch Labeler — {title} — Press Q to close"

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(window_name=win_title, width=960, height=720)
    vis.add_geometry(pcd)

    opt = vis.get_render_option()
    opt.background_color = COLOR_BG
    opt.point_size = POINT_SIZE

    vis.register_key_callback(ord("Q"), lambda v: v.close())
    vis.register_key_callback(ord("q"), lambda v: v.close())

    vis.run()
    vis.destroy_window()


# ── ラベル保存 ────────────────────────────────────────────────────────────

class LabelStore:
    """パッチラベルの読み書きを管理するクラス"""

    def __init__(
        self,
        save_path: Path,
        ply_name: str,
        n_fps: int,
        n_neighbors: int,
    ) -> None:
        self.path = save_path
        self.ply_name = ply_name
        self.n_fps = n_fps
        self.n_neighbors = n_neighbors
        self.data: dict[str, dict] = {}

        if save_path.exists():
            with open(save_path, encoding="utf-8") as f:
                loaded = json.load(f)
            self.data = loaded.get("patches", loaded)
            print(f"[再開] {len(self.data)} 件読み込み済み: {save_path}")

    def is_labeled(self, pid: int) -> bool:
        return str(pid) in self.data

    def get_names(self, pid: int) -> list[str]:
        return self.data.get(str(pid), {}).get("names", [])

    def get_status(self, pid: int) -> str:
        return self.data.get(str(pid), {}).get("status", "未ラベル")

    def set_label(
        self,
        patch_ids: list[int],
        names: list[str],
        centers: np.ndarray,
    ) -> None:
        for pid in patch_ids:
            self.data[str(pid)] = {
                "patch_id": pid,
                "names": names,
                "center_xyz": centers[pid].tolist(),
                "status": "labeled",
            }

    def set_skip(self, patch_ids: list[int], centers: np.ndarray) -> None:
        for pid in patch_ids:
            self.data[str(pid)] = {
                "patch_id": pid,
                "names": [],
                "center_xyz": centers[pid].tolist(),
                "status": "skipped",
            }

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        output = {
            "ply": self.ply_name,
            "n_fps": self.n_fps,
            "n_neighbors": self.n_neighbors,
            "patches": self.data,
        }
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(output, f, ensure_ascii=False, indent=2)
        n_done = sum(1 for v in self.data.values() if v.get("status") == "labeled")
        n_skip = sum(1 for v in self.data.values() if v.get("status") == "skipped")
        print(f"[保存] ラベル済: {n_done}  スキップ: {n_skip}  → {self.path}")

    def unlabeled_ids(self, total: int) -> list[int]:
        return [i for i in range(total) if not self.is_labeled(i)]


# ── コマンドループ ────────────────────────────────────────────────────────

def run_interactive(
    pts: np.ndarray,
    centers: np.ndarray,
    n_neighbors: int,
    store: LabelStore,
) -> None:
    n_patches = len(centers)

    unlabeled = store.unlabeled_ids(n_patches)
    if not unlabeled:
        print("全パッチにラベルが付いています。終了します。")
        return

    # 現在選択中のパッチ ID リスト
    selection: list[int] = [unlabeled[0]]

    def current_highlight() -> np.ndarray:
        return gather_patch_point_indices(pts, centers, selection, n_neighbors)

    def title_str() -> str:
        n_done = n_patches - len(store.unlabeled_ids(n_patches))
        return f"patch {selection} | {n_done}/{n_patches} 完了"

    def show_status() -> None:
        unlabeled_now = store.unlabeled_ids(n_patches)
        n_done = n_patches - len(unlabeled_now)
        print(f"\n{'─'*56}")
        print(f"  進捗: {n_done}/{n_patches} 完了  (残り {len(unlabeled_now)} 件)")
        print(f"  選択中: patch {selection}  ({len(selection)} パッチ)")
        for pid in selection:
            names = store.get_names(pid)
            lbl = "; ".join(names) if names else "(未ラベル)"
            c = centers[pid]
            print(f"    patch {pid:03d}: [{c[0]:+.3f}, {c[1]:+.3f}, {c[2]:+.3f}]  → {lbl}")
        print(f"{'─'*56}")

    def show_commands() -> None:
        print()
        print("  コマンド:")
        print("    vis              再可視化")
        print("    label <名前>     ラベル付け (複数: name1; name2; ...)")
        print("    add              最も近い未ラベルパッチを選択に追加")
        print("    remove           最も遠いパッチを選択から削除")
        print("    skip             選択パッチをスキップ")
        print("    next             選択を変えずに次の未ラベルへ移動")
        print("    list             全パッチ一覧")
        print("    info             現在の選択詳細")
        print("    save             保存")
        print("    quit             保存して終了")
        print()

    # ── 最初の可視化 ──
    show_status()
    print("\n[可視化中] Q を押すとコマンド入力モードへ...")
    visualize(pts, current_highlight(), title_str())

    # ── メインループ ──
    while True:
        show_status()
        show_commands()

        try:
            cmd_raw = input("  → ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n中断。保存します。")
            store.save()
            break

        if not cmd_raw:
            continue

        parts = cmd_raw.split(None, 1)
        cmd = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else ""

        # ── vis ──
        if cmd == "vis":
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── label ──
        elif cmd == "label":
            if not arg:
                print("  ⚠  名前を入力してください (例: label 下顎頭; 関節頭)")
                continue
            names = [n.strip() for n in arg.split(";") if n.strip()]
            if not names:
                print("  ⚠  有効な名前がありません")
                continue

            store.set_label(selection, names, centers)
            store.save()
            print(f"  ✓ patch {selection} → {names}")

            # 次の未ラベルへ自動移動
            unlabeled = store.unlabeled_ids(n_patches)
            if not unlabeled:
                print("\n  全パッチにラベルが付きました！")
                break
            selection = [unlabeled[0]]
            print(f"\n  → 次のパッチ: {selection}")
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── add ──
        elif cmd == "add":
            unlabeled = store.unlabeled_ids(n_patches)
            candidates = [i for i in unlabeled if i not in selection]
            if not candidates:
                print("  ⚠  追加できる未ラベルパッチがありません")
                continue
            # 最初に表示されたパッチ(selection[0])に最も近い候補を追加
            base_center = centers[selection[0]]
            cand_centers = centers[candidates]
            dists = np.sum((cand_centers - base_center) ** 2, axis=1)
            nearest = candidates[int(np.argmin(dists))]
            selection.append(nearest)
            print(f"  ✓ patch {nearest:03d} を追加 → 選択: {selection}")
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── remove ──
        elif cmd == "remove":
            if len(selection) <= 1:
                print("  ⚠  選択が 1 パッチ以下のため削除できません")
                continue
            sel_centers = centers[selection]
            # 最初に表示されたパッチ(selection[0])から最も遠いものを削除
            base_center = centers[selection[0]]
            dists = np.sum((sel_centers - base_center) ** 2, axis=1)
            farthest_idx = int(np.argmax(dists))
            removed = selection.pop(farthest_idx)
            print(f"  ✓ patch {removed:03d} を削除 → 選択: {selection}")
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── skip ──
        elif cmd == "skip":
            store.set_skip(selection, centers)
            store.save()
            print(f"  ✓ patch {selection} → スキップ")
            unlabeled = store.unlabeled_ids(n_patches)
            if not unlabeled:
                print("\n  全パッチ処理完了！")
                break
            selection = [unlabeled[0]]
            print(f"\n  → 次のパッチ: {selection}")
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── next ──
        elif cmd == "next":
            unlabeled = store.unlabeled_ids(n_patches)
            remaining = [i for i in unlabeled if i not in selection]
            if not remaining:
                print("  ⚠  次の未ラベルパッチがありません")
                continue
            selection = [remaining[0]]
            print(f"  → パッチ: {selection}")
            print("[可視化中] Q を押すとコマンド入力モードへ...")
            visualize(pts, current_highlight(), title_str())

        # ── list ──
        elif cmd == "list":
            print(f"\n  {'patch':>6}  {'状態':<10}  {'中心座標':<32}  名前")
            print(f"  {'─'*72}")
            for i in range(n_patches):
                status = store.get_status(i)
                names = store.get_names(i)
                names_str = "; ".join(names) if names else ""
                c = centers[i]
                coord = f"[{c[0]:+.3f},{c[1]:+.3f},{c[2]:+.3f}]"
                marker = "►" if i in selection else " "
                print(f"  {marker} {i:04d}  {status:<10}  {coord:<32}  {names_str}")

        # ── info ──
        elif cmd == "info":
            show_status()

        # ── save ──
        elif cmd == "save":
            store.save()

        # ── quit ──
        elif cmd in ("quit", "exit", "q"):
            store.save()
            print("  終了します。")
            break

        else:
            print(f"  ⚠  不明なコマンド: '{cmd_raw}'")
            print("     'vis', 'label <名前>', 'add', 'remove', 'skip',")
            print("     'next', 'list', 'info', 'save', 'quit' を使用してください")


# ── エントリポイント ──────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="下顎骨パッチ手動解剖構造ラベリングツール",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--ply", type=str, default=None,
        help="PLY ファイルパス (--case と排他)",
    )
    parser.add_argument(
        "--case", type=str, default=None,
        help="症例 ID (例: 'Pat 1a')。config.py の PLY_DIR から自動解決 (--ply と排他)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="利用可能な PLY 症例を一覧表示して終了",
    )
    parser.add_argument(
        "--n_sample", type=int, default=4096,
        metavar="N",
        help="PLY から初期サンプリングする点数 (default: 4096)",
    )
    parser.add_argument(
        "--n_fps", type=int, default=32,
        metavar="K",
        help="FPS で抽出する代表点数 = パッチ数 (default: 32)",
    )
    parser.add_argument(
        "--n_neighbors", type=int, default=64,
        metavar="M",
        help="各代表点周りに表示する近傍点数 (default: 64)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="ラベル保存先 JSON (default: manual/<ply_stem>_labels.json)",
    )
    args = parser.parse_args()

    # ── 症例一覧 (config が使える場合) ──
    ply_files: dict[str, Path] = {}
    try:
        import config as cfg
        ply_files = cfg.get_ply_files()
    except Exception:
        pass

    if args.list:
        if not ply_files:
            print("PLY ファイルが見つかりません。config.py の PLY_DIR を確認してください。")
        else:
            print("利用可能な症例:")
            for key, path in ply_files.items():
                print(f"  {key:<12}  {path}")
        sys.exit(0)

    # ── PLY パス解決 ──
    ply_path: Optional[Path] = None

    if args.ply and args.case:
        parser.error("--ply と --case は同時に指定できません")

    if args.case:
        if args.case not in ply_files:
            print(f"[ERROR] 症例 '{args.case}' が見つかりません。--list で確認してください。")
            sys.exit(1)
        ply_path = ply_files[args.case]
    elif args.ply:
        ply_path = Path(args.ply)
        if not ply_path.exists():
            print(f"[ERROR] PLY ファイルが見つかりません: {ply_path}")
            sys.exit(1)
    else:
        if ply_files:
            first_key = next(iter(ply_files))
            ply_path = ply_files[first_key]
            print(f"[INFO] PLY 未指定のため最初の症例を使用: {first_key} ({ply_path})")
        else:
            parser.error("--ply または --case を指定してください")

    # ── 出力パス ──
    output_path = Path(args.output) if args.output else _HERE / f"{ply_path.stem}_labels.json"

    ply_name = ply_path.stem

    print(f"\n{'='*60}")
    print(f"  PLY        : {ply_path.name}")
    print(f"  n_sample   : {args.n_sample}")
    print(f"  n_fps      : {args.n_fps}  (FPS 代表点 = パッチ数)")
    print(f"  n_neighbors: {args.n_neighbors}  (近傍点数)")
    print(f"  出力       : {output_path}")
    print(f"{'='*60}\n")

    # ── 点群 + FPS ──
    print("点群を読み込み中...")
    pts = load_pointcloud(ply_path, args.n_sample)
    print(f"  → {pts.shape}  座標範囲: [{pts.min():.3f}, {pts.max():.3f}]")

    print(f"FPS で {args.n_fps} 個の代表点を抽出中...")
    centers, _ = compute_fps_centers(pts, args.n_fps)
    print(f"  → centers: {centers.shape}")

    # ── ラベルストア ──
    store = LabelStore(output_path, ply_name, args.n_fps, args.n_neighbors)

    # ── インタラクティブラベリング ──
    run_interactive(pts, centers, args.n_neighbors, store)

    print("\nラベリング終了。")


if __name__ == "__main__":
    main()
