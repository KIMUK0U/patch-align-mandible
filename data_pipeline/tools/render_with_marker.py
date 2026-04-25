"""
render_with_marker.py — matplotlib ベースのオフスクリーンレンダリング

Open3D OffscreenRenderer は EGL に依存するため macOS では動作しない。
代わりに matplotlib (Agg バックエンド) を使用し、
macOS / Linux / Google Colab すべてで EGL なしに動作する。

パッチ領域の頂点を距離に応じたグラデーションで描画する:
  - 中心 (距離=0): PATCH_COLOR (オレンジ) で不透明・大きく描画
  - パッチ境界 (距離=patch_radius): BASE_COLOR (グレー青) に滑らかに遷移
  - パッチ外: BASE_COLOR で薄く描画
これにより LLM が「どこを中心としたパッチか」を視覚的に把握しやすくなる。
プロンプトで「色・グラデーションの記載をするな」と指示することで
LLM の出力から視覚表現情報を排除する。
"""

import io
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
import pyvista as pv

# matplotlib は import より前に backend を設定する必要がある
import matplotlib
matplotlib.use('Agg')  # ヘッドレス / EGL 不要バックエンド
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D projection の登録に必要)

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from tools.lps_utils import lps_octant, normalize_pointcloud

# ── ハイライトカラー設定 ──────────────────────────────────────────────────
PATCH_COLOR = [1.0, 0.45, 0.10]    # オレンジ (パッチ領域)
BASE_COLOR  = [0.75, 0.82, 0.87]   # グレー青 (非パッチ領域・骨の質感)

# ── matplotlib 10 視点 (elev°, azim°) ─────────────────────────────────────
# LPS 座標系: x=Left, y=Posterior, z=Superior
# matplotlib の規則: azim=0 → +x方向から観察, azim=90 → +y方向から観察
VIEWPOINTS: dict[str, dict] = {
    "anterior":  {"elev":   0, "azim": 270},  # -y 方向 (患者正面)
    "posterior": {"elev":   0, "azim":  90},  # +y 方向 (患者背面)
    "left":      {"elev":   0, "azim":   0},  # +x 方向 (患者左側)
    "right":     {"elev":   0, "azim": 180},  # -x 方向 (患者右側)
    "superior":  {"elev":  90, "azim":   0},  # +z 方向 (上方)
    "inferior":  {"elev": -90, "azim":   0},  # -z 方向 (下方)
    "az045":     {"elev":   0, "azim":  45},  # 左後斜め
    "az135":     {"elev":   0, "azim": 135},  # 右後斜め
    "az225":     {"elev":   0, "azim": 225},  # 右前斜め
    "az315":     {"elev":   0, "azim": 315},  # 左前斜め
}

# ── z 軸周り 8 方位 (CLIP モード用, elev=0, azim=0°〜315°, 45°刻り) ─────
# azim=0   → +x 方向 (患者左側)          azim=180 → -x 方向 (患者右側)
# azim=90  → +y 方向 (患者後方)          azim=270 → -y 方向 (患者正面)
# azim=45/135/225/315 → 斜め 4 方向
AZIMUTHAL_VIEWS: dict[str, dict] = {
    f"az{a:03d}": {"elev": 0, "azim": a}
    for a in range(0, 360, 45)
}
# az000=患者左, az045=左後斜め, az090=後方, az135=右後斜め,
# az180=患者右, az225=右前斜め, az270=前方, az315=左前斜め

# LLM に渡す視点優先順 (左右・正中で異なる)
VIEW_PRIORITY_LEFT    = ["anterior", "left", "superior", "posterior", "inferior", "az315", "az045", "right", "az225", "az135"]
VIEW_PRIORITY_RIGHT   = ["anterior", "right", "superior", "posterior", "inferior", "az225", "az135", "left", "az315", "az045"]
VIEW_PRIORITY_MIDLINE = ["anterior", "superior", "inferior", "posterior", "left", "right", "az315", "az045", "az225", "az135"]

MAX_SCATTER_PTS  = 8192       # 点群モード: 描画する最大頂点数
MAX_RENDER_FACES = 1_000_000  # STL メッシュモード: 描画する最大面数


# ── 内部ユーティリティ ────────────────────────────────────────────────────

def _get_normalized_vertices(
    mesh,
    max_pts: int = MAX_SCATTER_PTS,
) -> np.ndarray:
    """
    Open3D TriangleMesh から正規化済み頂点 (N, 3) を取得する。
    頂点数が max_pts を超える場合は均一サブサンプリングする。
    """
    verts = np.asarray(mesh.vertices, dtype=np.float32)
    verts_norm = normalize_pointcloud(verts)
    if len(verts_norm) > max_pts:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(verts_norm), max_pts, replace=False)
        verts_norm = verts_norm[idx]
    return verts_norm


# ── メインレンダリング関数 ────────────────────────────────────────────────

def render_mesh_with_patch_color(
    mesh,
    patch_center_lps: np.ndarray,
    viewpoint: dict,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    パッチ領域をハイライトした点群を matplotlib でレンダリングする。

    Args:
        mesh:             open3d TriangleMesh (任意スケール)
        patch_center_lps: (3,) 正規化 LPS 座標でのパッチ中心
        viewpoint:        VIEWPOINTS の 1 エントリ {"elev": ..., "azim": ...}
        patch_radius:     ハイライト半径 (正規化座標)
        image_size:       出力画像サイズ (W, H)

    Returns:
        img: (H, W, 3) uint8 RGB
    """
    verts_norm = _get_normalized_vertices(mesh)
    dists    = np.linalg.norm(verts_norm - patch_center_lps, axis=1)
    is_patch = dists <= patch_radius

    dpi = 100
    fig = plt.figure(
        figsize=(image_size[0] / dpi, image_size[1] / dpi),
        dpi=dpi,
    )
    ax = fig.add_subplot(111, projection='3d')
    fig.patch.set_facecolor('black')
    ax.set_facecolor('black')

    # 非パッチ点 (薄く・小さく)
    base_pts = verts_norm[~is_patch]
    if len(base_pts):
        ax.scatter(
            base_pts[:, 0], base_pts[:, 1], base_pts[:, 2],
            c=[BASE_COLOR], s=10, alpha=1.0, linewidths=0,
        )

    # パッチ点: 距離に応じたグラデーション
    # t=0 (中心) → PATCH_COLOR (オレンジ), t=1 (境界) → BASE_COLOR (グレー青)
    # 遠い点から順に描画し、中心点が最後 (前面) に重なるようにソート
    patch_idx  = np.where(is_patch)[0]
    if len(patch_idx):
        patch_dists = dists[patch_idx]
        patch_pts   = verts_norm[patch_idx]

        t = np.clip(patch_dists / patch_radius, 0.0, 1.0)  # (M,)

        # RGBA 色: 中心ほど PATCH_COLOR・不透明, 縁ほど BASE_COLOR
        pc = np.array(PATCH_COLOR, dtype=np.float32)
        bc = np.array(BASE_COLOR,  dtype=np.float32)
        rgb        = (1.0 - t)[:, None] * pc + t[:, None] * bc    # (M, 3)
        alpha_vals = np.ones_like(t)                                # alpha = 1.0
        rgba = np.concatenate([rgb, alpha_vals[:, None]], axis=1)  # (M, 4)

        # 点サイズ: 中心 s=12, 縁 s=2
        sizes = 10.0 + 2.0 * (1.0 - t)

        # 遠い順 → 近い順にソートして中心が前面に来るようにする
        order      = np.argsort(patch_dists)[::-1]
        patch_pts  = patch_pts[order]
        rgba       = rgba[order]
        sizes      = sizes[order]

        ax.scatter(
            patch_pts[:, 0], patch_pts[:, 1], patch_pts[:, 2],
            c=rgba, s=sizes, linewidths=0,
        )

    ax.view_init(elev=viewpoint['elev'], azim=viewpoint['azim'])
    ax.set_box_aspect([1, 1, 1])
    ax.set_xlim(-1, 1)
    ax.set_ylim(-1, 1)
    ax.set_zlim(-1, 1)
    ax.axis('off')
    plt.tight_layout(pad=0)

    buf = io.BytesIO()
    fig.savefig(
        buf, format='png', dpi=dpi,
        bbox_inches='tight',
        facecolor='black', edgecolor='none',
    )
    plt.close(fig)
    buf.seek(0)

    img = np.array(Image.open(buf).convert('RGB').resize(image_size))
    return img.astype(np.uint8)


def annotate_image(
    img: np.ndarray,
    patch_id: int,
    octant: str,
    view_name: str,
) -> np.ndarray:
    """左上に patch_id, LPS オクタント, 視点名を描画する"""
    out = img.copy()
    cv2.putText(out, f"Patch #{patch_id}",  (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220,  20,  20), 2)
    cv2.putText(out, f"LPS: {octant}",       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, ( 20,  20, 200), 2)
    cv2.putText(out, f"View: {view_name}",   (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, ( 50, 150,  50), 2)
    return out


def render_all_views(
    mesh,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, np.ndarray]:
    """全視点レンダリング。PyVista/VTK オフスクリーン (Phong シェーディング)。

    Returns: {"anterior": img_np, "posterior": img_np, ...}
    """
    return render_all_views_pyvista(
        mesh, patch_center_lps, patch_id, patch_radius, image_size,
    )


def render_azimuthal_views(
    mesh,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, np.ndarray]:
    """z 軸周り 8 方位レンダリング (CLIP モード用)。PyVista/VTK オフスクリーン。

    Returns: {"az000": img_np, "az045": img_np, ..., "az315": img_np}
    """
    return render_azimuthal_views_pyvista(
        mesh, patch_center_lps, patch_id, patch_radius, image_size,
    )


def select_views_for_llm(
    rendered: dict[str, np.ndarray],
    octant: str,
    max_views: int = 10,
) -> list[str]:
    """LLM に渡す視点を最大 max_views 枚選ぶ (左右・正中に応じた優先順)"""
    if octant.startswith("L"):
        priority = VIEW_PRIORITY_LEFT
    elif octant.startswith("R"):
        priority = VIEW_PRIORITY_RIGHT
    else:  # "M-" 正中
        priority = VIEW_PRIORITY_MIDLINE
    return [v for v in priority if v in rendered][:max_views]


def image_to_base64(img_np: np.ndarray) -> str:
    """numpy RGB 画像 → base64 エンコード PNG 文字列"""
    import base64
    buf = io.BytesIO()
    Image.fromarray(img_np).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def render_mesh_triangles_with_patch_color(
    stl_mesh,
    verts_norm: np.ndarray,
    patch_center_lps: np.ndarray,
    viewpoint: dict,
    patch_radius: float = 0.15,
    max_faces: int = MAX_RENDER_FACES,
    image_size: tuple[int, int] = (512, 512),
) -> np.ndarray:
    """
    STL メッシュの三角形面を cv2 オキュパンシーレンダラーで描画する。

    バックフェースカリング → 手前の面から順に cv2.fillPoly で描画し、
    占有済みピクセルには後続の面を上書きしない (ソフトウェア Z バッファ)。
    matplotlib Poly3DCollection の内部深度ソート問題を回避し、
    非凸メッシュでも透けが発生しない。

    Args:
        stl_mesh:          open3d TriangleMesh (STL)
        verts_norm:        (V, 3) STL 頂点を独立して正規化した座標 ([-1, 1] スケール)
        patch_center_lps:  (3,) 正規化 LPS 座標でのパッチ中心
        viewpoint:         VIEWPOINTS の 1 エントリ {"elev": ..., "azim": ...}
        patch_radius:      ハイライト半径 (正規化座標)
        max_faces:         描画する最大面数 (超過時はランダムサブサンプリング)
        image_size:        出力画像サイズ (W, H)

    Returns:
        img: (H, W, 3) uint8 RGB (黒背景)
    """
    W, H = image_size

    tris = np.asarray(stl_mesh.triangles)  # (F, 3)
    if len(tris) > max_faces:
        rng = np.random.default_rng(42)
        tris = tris[rng.choice(len(tris), max_faces, replace=False)]

    face_verts = verts_norm[tris].astype(np.float64)  # (F, 3, 3)

    # NaN/Inf を含む面を除外 (corrupt な STL 面や float32 オーバーフロー対策)
    finite_mask = np.all(np.isfinite(face_verts.reshape(len(face_verts), -1)), axis=1)
    face_verts  = face_verts[finite_mask]
    if len(face_verts) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # ── カメラ基底ベクトルを構築 ─────────────────────────────────────────
    azim_rad = np.deg2rad(viewpoint['azim'])
    elev_rad = np.deg2rad(viewpoint['elev'])
    cam_fwd  = np.array([
        np.cos(elev_rad) * np.cos(azim_rad),
        np.cos(elev_rad) * np.sin(azim_rad),
        np.sin(elev_rad),
    ], dtype=np.float64)
    world_up = np.array([0., 0., 1.], dtype=np.float64)
    if abs(np.dot(cam_fwd, world_up)) > 0.99:
        world_up = np.array([1., 0., 0.], dtype=np.float64)
    right = np.cross(world_up, cam_fwd)
    right /= np.linalg.norm(right)
    up    = np.cross(cam_fwd, right)
    up   /= np.linalg.norm(up)

    # ── バックフェースカリング ────────────────────────────────────────────
    # np.einsum を使用して macOS Accelerate BLAS の誤 FP 例外警告を回避する
    e1      = face_verts[:, 1] - face_verts[:, 0]
    e2      = face_verts[:, 2] - face_verts[:, 0]
    normals = np.cross(e1, e2)
    face_verts = face_verts[np.einsum('ij,j->i', normals, cam_fwd) > 0]
    if len(face_verts) == 0:
        return np.zeros((H, W, 3), dtype=np.uint8)

    # ── 直交投影 (スクリーン座標) ─────────────────────────────────────────
    scale = 1.1
    vf    = face_verts.reshape(-1, 3)           # (F'*3, 3)
    px    = ((np.einsum('ij,j->i', vf, right) / scale + 1.) * 0.5 * W).reshape(-1, 3).astype(np.float32)
    py    = ((1. - np.einsum('ij,j->i', vf, up) / scale)    * 0.5 * H).reshape(-1, 3).astype(np.float32)
    z_dep = np.einsum('ij,j->i', vf, cam_fwd).reshape(-1, 3)            # (F', 3) 大きいほど手前

    # ── パッチ着色 ────────────────────────────────────────────────────────
    centroids = face_verts.mean(axis=1)
    t = np.clip(
        np.linalg.norm(centroids - patch_center_lps, axis=1) / patch_radius,
        0., 1.,
    )
    pc     = np.array(PATCH_COLOR, dtype=np.float32)
    bc     = np.array(BASE_COLOR,  dtype=np.float32)
    rgb_u8 = (((1. - t[:, None]) * pc + t[:, None] * bc) * 255).astype(np.uint8)

    # ── 手前→奥ソート (手前面を先に描画し、奥面は占有済みピクセルをスキップ) ─
    order  = np.argsort(z_dep.mean(axis=1))[::-1]  # 大 (手前) → 小 (奥)
    px     = px[order]
    py     = py[order]
    rgb_u8 = rgb_u8[order]

    # ── cv2 オキュパンシーマスクレンダリング ─────────────────────────────
    img      = np.zeros((H, W, 3), dtype=np.uint8)
    occupied = np.zeros((H, W),    dtype=np.uint8)
    tri_mask = np.zeros((H, W),    dtype=np.uint8)

    for i in range(len(px)):
        pts = np.stack(
            [px[i].round(), py[i].round()], axis=1,
        ).astype(np.int32)
        tri_mask.fill(0)
        cv2.fillPoly(tri_mask, [pts], 1)
        draw = (tri_mask == 1) & (occupied == 0)
        if draw.any():
            img[draw]      = rgb_u8[i]
            occupied[draw] = 1

    return img


def render_all_views_stl(
    stl_mesh,
    verts_norm: np.ndarray,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    max_faces: int = MAX_RENDER_FACES,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, np.ndarray]:
    """STL メッシュの 10 視点レンダリング。PyVista/VTK オフスクリーン (Phong シェーディング)。

    verts_norm 引数は後方互換のため残存するが PyVista 実装では不使用
    (内部で normalize_pointcloud を再計算)。

    Returns: {"anterior": img_np, "posterior": img_np, ...}
    """
    return render_all_views_pyvista(
        stl_mesh, patch_center_lps, patch_id, patch_radius, image_size,
    )


# ── PyVista/VTK ベースの高速レンダラー ────────────────────────────────────
# matplotlib Agg (~200ms/枚) の代替。VTK オフスクリーン OpenGL を使用し、
# 1 パッチにつき 1 プロッタを生成して全視点を連続スクリーンショット。
# → Phong シェーディング + 3 点照明で立体感と陰影を実現。
# macOS では EGL 不要 (VTK ネイティブ Framebuffer を使用)。

def _build_pyvista_mesh(
    mesh,
    patch_center_lps: np.ndarray,
    patch_radius: float,
) -> pv.PolyData:
    """Open3D TriangleMesh → PyVista PolyData + 頂点カラー。
    三角形なし (点群のみ) の場合は頂点だけの PolyData を返す。"""
    verts = normalize_pointcloud(np.asarray(mesh.vertices, dtype=np.float32))
    tris  = np.asarray(mesh.triangles, dtype=np.int32)

    dists = np.linalg.norm(verts - patch_center_lps, axis=1)
    t     = np.clip(dists / patch_radius, 0.0, 1.0)
    pc    = np.array(PATCH_COLOR, dtype=np.float32)
    bc    = np.array(BASE_COLOR,  dtype=np.float32)
    vertex_rgb = (1.0 - t[:, None]) * pc + t[:, None] * bc  # (N, 3) float32

    if len(tris) > 0:
        pv_faces = np.hstack(
            [np.full((len(tris), 1), 3, dtype=np.int32), tris]
        ).ravel()
        poly = pv.PolyData(verts.astype(float), pv_faces)
        poly = poly.compute_normals(auto_orient_normals=True, consistent_normals=True)
    else:
        poly = pv.PolyData(verts.astype(float))

    poly['RGB'] = (vertex_rgb * 255.0).clip(0, 255).astype(np.uint8)
    return poly


def _viewpoint_to_camera_pos(viewpoint: dict, r: float = 5.0) -> tuple:
    """matplotlib elev/azim → PyVista カメラ位置 (cam_fwd 式と同一)。"""
    azim_rad = np.radians(viewpoint['azim'])
    elev_rad = np.radians(viewpoint['elev'])
    cx = r * np.cos(elev_rad) * np.cos(azim_rad)
    cy = r * np.cos(elev_rad) * np.sin(azim_rad)
    cz = r * np.sin(elev_rad)
    return (cx, cy, cz)


def render_all_views_pyvista(
    mesh,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
    viewpoints: dict | None = None,
) -> dict[str, np.ndarray]:
    """PyVista/VTK オフスクリーンで全視点をレンダリングする。

    1 パッチにつき 1 プロッタを生成し全視点を連続スクリーンショット。
    Phong シェーディング + 3 点照明 (キー/フィル/バック) で立体感を実現。

    Returns: {"anterior": img_np, ...}  (annotate_image 適用済み)
    """
    if viewpoints is None:
        viewpoints = VIEWPOINTS

    poly   = _build_pyvista_mesh(mesh, patch_center_lps, patch_radius)
    octant = lps_octant(patch_center_lps)

    pl = pv.Plotter(off_screen=True, window_size=list(image_size))
    pl.set_background('black')

    pl.remove_all_lights()
    pl.add_light(pv.Light(position=( 2.0,  1.0,  3.0), intensity=0.75))  # キー光
    pl.add_light(pv.Light(position=(-2.0,  1.0,  2.0), intensity=0.45))  # フィル光
    pl.add_light(pv.Light(position=( 0.0, -3.0,  1.0), intensity=0.20))  # バック光

    mesh_kwargs: dict = dict(
        scalars='RGB', rgb=True,
        smooth_shading=(len(np.asarray(mesh.triangles)) > 0),
        ambient=0.25, diffuse=0.65, specular=0.10, specular_power=20,
    )
    if len(np.asarray(mesh.triangles)) == 0:
        mesh_kwargs.update(point_size=5, render_points_as_spheres=True)

    pl.add_mesh(poly, **mesh_kwargs)

    rendered: dict[str, np.ndarray] = {}
    for view_name, vp in viewpoints.items():
        pl.camera.position    = _viewpoint_to_camera_pos(vp)
        pl.camera.focal_point = (0.0, 0.0, 0.0)
        pl.camera.up = (1.0, 0.0, 0.0) if abs(vp['elev']) > 80 else (0.0, 0.0, 1.0)
        pl.camera.reset_clipping_range()
        pl.render()                                      # カメラ変更をオフスクリーンに確定
        img = pl.screenshot(return_img=True)             # (H, W, 3) uint8
        if img.shape[:2] != (image_size[1], image_size[0]):
            img = cv2.resize(img, image_size)
        rendered[view_name] = annotate_image(img, patch_id, octant, view_name)

    pl.close()
    return rendered


def render_azimuthal_views_pyvista(
    mesh,
    patch_center_lps: np.ndarray,
    patch_id: int,
    patch_radius: float = 0.15,
    image_size: tuple[int, int] = (512, 512),
) -> dict[str, np.ndarray]:
    """AZIMUTHAL_VIEWS (8 方位) を PyVista でレンダリング (CLIP モード用)。"""
    return render_all_views_pyvista(
        mesh, patch_center_lps, patch_id, patch_radius, image_size,
        viewpoints=AZIMUTHAL_VIEWS,
    )


def save_rendered_images(
    rendered: dict[str, np.ndarray],
    out_dir: Path,
) -> None:
    """6 視点画像を out_dir に保存する"""
    out_dir.mkdir(parents=True, exist_ok=True)
    for view_name, img in rendered.items():
        Image.fromarray(img).save(out_dir / f"{view_name}.png")


if __name__ == "__main__":
    import argparse
    import open3d as o3d

    parser = argparse.ArgumentParser(description="レンダリングテスト (単一パッチ)")
    parser.add_argument("ply",       help="PLY ファイルパス")
    parser.add_argument("--out",     default="/tmp/render_test", help="出力ディレクトリ")
    parser.add_argument("--radius",  type=float, default=0.15,   help="ハイライト半径")
    args = parser.parse_args()

    mesh = o3d.io.read_triangle_mesh(args.ply)
    # 後上方をテスト中心とする (正規化済み LPS 座標)
    center = np.array([0.0, 0.3, 0.4], dtype=np.float32)

    rendered = render_all_views(mesh, center, patch_id=0, patch_radius=args.radius)
    out_dir  = Path(args.out)
    save_rendered_images(rendered, out_dir)
    print(f"Saved {len(rendered)} images to {out_dir}")
