"""Stage 3 モデルラッパー。

Student モデル (学習対象) と Teacher モデル (固定) を統合する。
投影ヘッド PatchToTextProj も含む。

使用方法:
    model_s, model_t, proj = build_stage3_models(cfg, device, ckpt_path)
"""

import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict

# PatchAlign3D リポジトリを参照するためにパスを追加
# parents[0]=phase3_models/, parents[1]=Phase3_Training/, parents[2]=DentalPatchAligned3D/,
# parents[3]=ULIP_PointLLM/  ← _reference_repos/ はここにある
import types as _types
_REF_ROOT = Path(__file__).parents[3] / "_reference_repos" / "PatchAlign3D" / "src"
if str(_REF_ROOT) not in sys.path:
    sys.path.insert(0, str(_REF_ROOT))

# patchalign3d を仮想パッケージとして登録 (train スクリプト未経由で直接インポートされた場合に備えて)
if "patchalign3d" not in sys.modules:
    _pa3d = _types.ModuleType("patchalign3d")
    _pa3d.__path__ = [str(_REF_ROOT)]
    sys.modules["patchalign3d"] = _pa3d

from patchalign3d.models import point_transformer  # type: ignore


# ---------------------------------------------------------------------------
# 投影ヘッド (stage2.py の PatchToTextProj と同一)
# ---------------------------------------------------------------------------

class PatchToTextProj(nn.Module):
    """パッチ特徴 (B, trans_dim, G) → CLIP 空間 (B, G, clip_dim)。

    stage2.py の PatchToTextProj(384→1280) と完全互換。
    """

    def __init__(self, in_dim: int = 384, out_dim: int = 1280):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)

    def forward(self, patch_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_emb: (B, in_dim, G)
        Returns:
            (B, G, out_dim) 正規化済み
        """
        x = patch_emb.transpose(1, 2)      # (B, G, in_dim)
        x = self.proj(x)                   # (B, G, out_dim)
        return F.normalize(x, dim=-1)


# ---------------------------------------------------------------------------
# 温度モジュール (stage2.py の LearnableTemp と同一)
# ---------------------------------------------------------------------------

class LearnableTemp(nn.Module):
    def __init__(self, init_tau: float = 0.07, max_scale: float = 100.0):
        super().__init__()
        init_scale = 1.0 / max(init_tau, 1e-6)
        self.log_scale = nn.Parameter(
            torch.log(torch.tensor(init_scale, dtype=torch.float32))
        )
        self.max_scale = float(max_scale)

    def forward(self) -> torch.Tensor:
        return self.log_scale.exp().clamp(max=self.max_scale)


# ---------------------------------------------------------------------------
# モデル構築
# ---------------------------------------------------------------------------

def _build_encoder(cfg: EasyDict) -> nn.Module:
    """EasyDict config から point_transformer モデルを構築。"""
    model_cfg = EasyDict(
        trans_dim=cfg.get("trans_dim", 384),
        depth=cfg.get("depth", 12),
        drop_path_rate=cfg.get("drop_path_rate", 0.1),
        cls_dim=cfg.get("cls_dim", 50),
        num_heads=cfg.get("num_heads", 6),
        group_size=cfg.get("group_size", 32),
        num_group=cfg.get("num_group", 128),
        encoder_dims=cfg.get("encoder_dims", 256),
        color=cfg.get("color", False),
        num_classes=cfg.get("num_classes", 16),
    )
    return point_transformer.get_model(model_cfg)


def _load_ckpt(model: nn.Module, proj: nn.Module, path: str, device: torch.device) -> None:
    """Stage 2 チェックポイントをロードする。存在しなければスキップ。"""
    if not path:
        return
    p = Path(path)
    if not p.exists():
        print(f"[Stage3] チェックポイントが見つかりません (スキップ): {path}")
        return

    ckpt = torch.load(str(p), map_location="cpu", weights_only=False)

    # チェックポイント形式の吸収
    model_state = ckpt.get("model", ckpt.get("model_state_dict", None))
    proj_state  = ckpt.get("proj",  ckpt.get("proj_state_dict", None))

    if model_state is not None:
        # 'module.' プレフィックスを除去 (DataParallel 対応)
        model_state = OrderedDict(
            (k.replace("module.", ""), v) for k, v in model_state.items()
        )
        missing, unexpected = model.load_state_dict(model_state, strict=False)
        if missing:
            print(f"[Stage3] model missing keys ({len(missing)}): {missing[:5]}")
        if unexpected:
            print(f"[Stage3] model unexpected keys ({len(unexpected)}): {unexpected[:5]}")

    if proj_state is not None and proj is not None:
        proj_state = OrderedDict(
            (k.replace("module.", ""), v) for k, v in proj_state.items()
        )
        proj.load_state_dict(proj_state, strict=False)

    print(f"[Stage3] チェックポイントをロード: {path}")


def freeze_encoder_except_last_block(model: nn.Module) -> None:
    """エンコーダの最終 Transformer ブロック以外をすべて凍結する。

    stage2.py の freeze_encoder_except_last_block と同一ロジック。
    """
    for p in model.parameters():
        p.requires_grad = False

    # 最終 Transformer ブロックを解凍
    last_block = None
    if hasattr(model, "blocks"):
        inner = getattr(model.blocks, "blocks", None)
        if inner and len(inner) > 0:
            last_block = inner[-1]
    if last_block is not None:
        for p in last_block.parameters():
            p.requires_grad = True

    for attr in ("norm", "reduce_dim"):
        if hasattr(model, attr):
            for p in getattr(model, attr).parameters():
                p.requires_grad = True
    for attr in ("cls_token", "cls_pos"):
        if hasattr(model, attr):
            getattr(model, attr).requires_grad = True


def build_stage3_models(
    cfg: EasyDict,
    device: torch.device,
    ckpt_path: str = "",
) -> tuple[nn.Module, nn.Module, PatchToTextProj, LearnableTemp]:
    """Student・Teacher・ProjHead・温度モジュールを構築して返す。

    Parameters
    ----------
    cfg        : 設定 EasyDict
    device     : torch.device
    ckpt_path  : Stage 2 チェックポイントパス (空文字でスキップ)

    Returns
    -------
    (student, teacher, proj, temp)
      student : 学習対象 (最終ブロックのみ解凍)
      teacher : 固定コピー (eval モード、requires_grad=False)
      proj    : PatchToTextProj (全パラメータ学習可能)
      temp    : LearnableTemp
    """
    trans_dim = cfg.get("trans_dim", 384)
    clip_dim  = cfg.get("clip_dim",  1280)

    # Student
    student = _build_encoder(cfg).to(device)
    proj    = PatchToTextProj(in_dim=trans_dim, out_dim=clip_dim).to(device)
    temp    = LearnableTemp(init_tau=cfg.get("clip_tau", 0.07)).to(device)

    # チェックポイントから重みをロード
    _load_ckpt(student, proj, ckpt_path, device)

    # Teacher: Student と同じ重みで初期化し、以降は固定
    teacher = _build_encoder(cfg).to(device)
    teacher.load_state_dict(student.state_dict(), strict=True)
    for p in teacher.parameters():
        p.requires_grad = False
    teacher.eval()

    # 凍結戦略: 最終ブロック以外を凍結
    freeze_encoder_except_last_block(student)

    n_train = sum(p.numel() for p in student.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in student.parameters())
    print(
        f"[Stage3] Student: 学習可能 {n_train:,} / 総 {n_total:,} パラメータ"
        f"  Proj: {sum(p.numel() for p in proj.parameters()):,}"
    )

    return student, teacher, proj, temp
