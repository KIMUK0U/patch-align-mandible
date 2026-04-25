"""DentalPatchDataset: PLY 点群 + candidates.json アノテーションを読み込む。

Phase 3 Stage 3a / Stage 3b 学習用データセット。

データソース:
  PLY : Dataset/6167726/STLs/PLYs/Pat Xa_sample.ply  (8192 点)
  JSON: data/DentalPatchData/dental_vocab/candidates.json
        {patient_id: [{patch_id, patch_center_lps, lps_octant, texts}, ...]}

各サンプルの返却形式 (stage2.py パイプライン互換):
  points       : Tensor (3, N=2048)  - 正規化 xyz、チャネルファースト
  point_labels : Tensor (N,)         - Stage 3a: 領域 ID / Stage 3b: パッチ ID
  label_masks  : Tensor (K, N)       - ラベル単位バイナリマスク
  label_names  : list[str]           - 各ラベルのテキスト
  item_id      : str
"""

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# LPS 8方位 → 整数領域 ID (Stage 3a 用)
OCTANT_TO_ID: dict[str, int] = {
    "L-A-I": 0, "L-A-S": 1, "L-P-I": 2, "L-P-S": 3,
    "M-A-I": 4,
    "R-A-I": 5, "R-A-S": 6, "R-P-I": 7, "R-P-S": 8,
}
# 各 octant のテキスト (Stage 3a の label_names として使用)
REGION_TEXTS: dict[int, str] = {
    0: "left anterior inferior mandible",
    1: "left anterior superior mandible",
    2: "left posterior inferior mandible",
    3: "left posterior superior mandible",
    4: "anterior inferior mandible",
    5: "right anterior inferior mandible",
    6: "right anterior superior mandible",
    7: "right posterior inferior mandible",
    8: "right posterior superior mandible",
}
NUM_OCTANTS = len(OCTANT_TO_ID)


# ---------------------------------------------------------------------------
# ヘルパー関数
# ---------------------------------------------------------------------------

def _fps_cpu(xyz: torch.Tensor, k: int, seed: int = 42) -> torch.Tensor:
    """CPU 最遠点サンプリング。インデックス (k,) を返す。"""
    gen = torch.Generator().manual_seed(seed)
    N = xyz.shape[0]
    if k >= N:
        return torch.arange(N, dtype=torch.long)
    idx = torch.zeros(k, dtype=torch.long)
    start = int(torch.randint(0, N, (1,), generator=gen).item())
    idx[0] = start
    dist = torch.full((N,), float("inf"))
    for i in range(1, k):
        p = xyz[idx[i - 1]]
        d = ((xyz - p) ** 2).sum(dim=1)
        dist = torch.minimum(dist, d)
        idx[i] = int(dist.argmax().item())
    return idx


def _load_ply(ply_path: Path) -> torch.Tensor:
    """PLY ファイルを読み込み (N, 3) float32 Tensor を返す。
    plyfile または open3d のいずれかが必要。
    """
    try:
        from plyfile import PlyData  # type: ignore
        ply = PlyData.read(str(ply_path))
        v = ply["vertex"]
        xyz = torch.tensor(
            np.stack([np.array(v["x"]), np.array(v["y"]), np.array(v["z"])], axis=1),
            dtype=torch.float32,
        )
        return xyz
    except ImportError:
        pass

    try:
        import open3d as o3d  # type: ignore
        pcd = o3d.io.read_point_cloud(str(ply_path))
        xyz = torch.tensor(np.asarray(pcd.points), dtype=torch.float32)
        return xyz
    except ImportError:
        pass

    raise RuntimeError(
        f"PLY ファイルを読み込めません。'plyfile' または 'open3d' をインストールしてください: {ply_path}"
    )


def _normalize_unit_sphere(xyz: torch.Tensor):
    """重心で中心化し、単位球に正規化する。
    Returns: (xyz_norm, centroid, scale)
    """
    centroid = xyz.mean(dim=0)
    xyz_c = xyz - centroid
    scale = xyz_c.norm(dim=1).max().clamp_min(1e-6)
    return xyz_c / scale, centroid, scale


# ---------------------------------------------------------------------------
# データセットクラス
# ---------------------------------------------------------------------------

class DentalPatchDataset(Dataset):
    """
    PatchAlign3D Phase 3 学習用歯科点群データセット。

    Parameters
    ----------
    ply_dir : str
        PLY ファイルが置かれたディレクトリ (例: Dataset/6167726/STLs/STLs/normPLYs/)
    json_path : str
        candidates.json へのパス
    npoints : int
        ダウンサンプリング後の点数 (デフォルト 2048)
    mode : str
        "3a" (octant 領域、InfoNCE 用) または "3b" (パッチテキスト、BCE 用)
    split : str
        "train" または "val"
    val_ratio : float
        バリデーションに使う患者の割合
    seed : int
        FPS・テキストサンプリング用シード
    text_augment : bool
        True のとき各エポックでランダムに 1 テキストを選択 (戦略 A)
        False のとき texts[0] を固定使用
    """

    def __init__(
        self,
        ply_dir: str,
        json_path: str,
        npoints: int = 2048,
        mode: str = "3b",
        split: str = "train",
        val_ratio: float = 0.2,
        seed: int = 42,
        text_augment: bool = True,
        point_augment: bool = False,
    ):
        super().__init__()
        self.ply_dir = Path(ply_dir)
        self.npoints = npoints
        self.mode = mode
        self.text_augment = text_augment
        self.point_augment = point_augment and (split == "train")
        self.seed = seed

        with open(json_path, "r", encoding="utf-8") as f:
            candidates: dict = json.load(f)

        # PLY が存在する患者のみ残す
        all_patients = sorted(candidates.keys())
        valid: list[str] = []
        for pat in all_patients:
            if (self.ply_dir / f"{pat}.ply").exists():
                valid.append(pat)

        if not valid:
            raise RuntimeError(
                f"{self.ply_dir} に有効な PLY ファイルが見つかりません。"
                " 'Pat Xa_norm.ply' という形式のファイルを確認してください。"
            )

        # 決定的 train/val 分割
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(valid))
        n_val = max(1, int(len(valid) * val_ratio))
        val_set = set(perm[:n_val].tolist())
        train_set = set(perm[n_val:].tolist())

        chosen = val_set if split == "val" else train_set
        self.patients: list[str] = [valid[i] for i in sorted(chosen)]
        self.candidates: dict = {p: candidates[p] for p in self.patients}

        if not self.patients:
            raise RuntimeError(f"split={split} のサンプルがありません")

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.patients)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        pat = self.patients[idx]
        patches: list[dict] = self.candidates[pat]
        ply_path = self.ply_dir / f"{pat}.ply"

        # ---- 点群ロード & FPS ダウンサンプリング ----
        xyz_raw = _load_ply(ply_path)                         # (N_raw, 3)
        fps_idx = _fps_cpu(xyz_raw, self.npoints, seed=self.seed + idx)
        xyz = xyz_raw[fps_idx]                                # (2048, 3)

        # ---- 単位球正規化 ----
        xyz_norm, centroid, scale = _normalize_unit_sphere(xyz)   # (2048, 3)

        # ---- アノテーションパッチ中心の正規化 (PLY と同じ変換) ----
        raw_centers = torch.tensor(
            [p["patch_center_lps"] for p in patches], dtype=torch.float32
        )  # (K_ann, 3)
        norm_centers = (raw_centers - centroid) / scale       # (K_ann, 3)

        # ---- 点群データ拡張（訓練時のみ）----
        if self.point_augment:
            from phase3_datasets.point_augment import augment_dental
            xyz_norm, norm_centers = augment_dental(xyz_norm, norm_centers)

        K_ann = len(patches)

        # ---- 各点を最近傍アノテーションパッチに割り当て ----
        # dists: (npoints, K_ann)
        dists = torch.cdist(
            xyz_norm.unsqueeze(0), norm_centers.unsqueeze(0)
        ).squeeze(0)
        nearest = dists.argmin(dim=1)                         # (npoints,)

        # ---- バイナリマスク (K_ann, npoints) ----
        label_masks_ann = torch.zeros(K_ann, self.npoints, dtype=torch.bool)
        for k in range(K_ann):
            label_masks_ann[k] = nearest == k

        # ---- モード別ラベル構築 ----
        if self.mode == "3a":
            label_masks_out, label_names_out, point_labels = self._build_3a(
                patches, label_masks_ann
            )
        else:
            label_masks_out, label_names_out, point_labels = self._build_3b(
                patches, label_masks_ann
            )

        # ---- チャネルファースト (3, N) ----
        points = xyz_norm.T.contiguous()  # (3, 2048)

        return {
            "points":       points,
            "point_labels": point_labels,
            "label_masks":  label_masks_out,
            "label_names":  label_names_out,
            "item_id":      pat,
        }

    # ------------------------------------------------------------------
    def _build_3a(
        self,
        patches: list[dict],
        label_masks_ann: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Stage 3a: octant 単位に統合した粗い領域マスクを返す。

        Returns: label_masks (R, N), label_names (R,), point_labels (N,)
        """
        region_masks = torch.zeros(NUM_OCTANTS, self.npoints, dtype=torch.bool)
        for k, patch in enumerate(patches):
            oid = OCTANT_TO_ID.get(patch["lps_octant"], 0)
            region_masks[oid] |= label_masks_ann[k]

        active = [i for i in range(NUM_OCTANTS) if region_masks[i].any()]
        label_masks_out = region_masks[active]                 # (R, N)
        label_names_out = [REGION_TEXTS[i] for i in active]

        point_labels = torch.full((self.npoints,), -1, dtype=torch.long)
        for new_i, old_i in enumerate(active):
            point_labels[region_masks[old_i]] = new_i

        return label_masks_out, label_names_out, point_labels

    def _build_3b(
        self,
        patches: list[dict],
        label_masks_ann: torch.Tensor,
    ) -> tuple[torch.Tensor, list[str], torch.Tensor]:
        """Stage 3b: パッチ単位のテキストラベルを返す。

        text_augment=True の場合、各パッチから 1 テキストをランダム選択。

        Returns: label_masks (K, N), label_names (K,), point_labels (N,)
        """
        if self.text_augment:
            label_names_out = [random.choice(p["texts"]) for p in patches]
        else:
            label_names_out = [p["texts"][0] for p in patches]

        point_labels = torch.full((self.npoints,), -1, dtype=torch.long)
        for k in range(len(patches)):
            point_labels[label_masks_ann[k]] = k

        return label_masks_ann, label_names_out, point_labels


# ---------------------------------------------------------------------------
# collate 関数 (stage2.py の collate_trainset と同様のリスト形式)
# ---------------------------------------------------------------------------

def collate_dental(batch: list[dict]) -> list[dict]:
    """サンプルのリストをそのまま返す (バッチ内で可変長マスクを扱うため)。"""
    return batch
