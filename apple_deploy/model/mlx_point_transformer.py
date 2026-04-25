"""MLX reimplementation of PatchAlign3D Point Transformer for inference.

Weights are loaded from a numpy npz produced by convert_weights.py.
No training ops: DropPath → identity, BatchNorm uses running statistics.

Architecture matches _reference_repos/PatchAlign3D/src/models/point_transformer.py
and Phase3_Training/phase3_models/stage3_model.py exactly.
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# BatchNorm (inference-only: uses running statistics)
# ---------------------------------------------------------------------------

class _BatchNorm(nn.Module):
    def __init__(self, num_features: int, eps: float = 1e-5):
        super().__init__()
        self.weight       = mx.ones(num_features)
        self.bias         = mx.zeros(num_features)
        self.running_mean = mx.zeros(num_features)
        self.running_var  = mx.ones(num_features)
        self.eps          = eps

    def __call__(self, x: mx.array) -> mx.array:
        scale = self.weight / mx.sqrt(self.running_var + self.eps)
        shift = self.bias - self.running_mean * scale
        return x * scale + shift


# ---------------------------------------------------------------------------
# Local patch encoder  (G, M, 3) -> (G, encoder_dims)
# Conv1d(kernel=1) is equivalent to Linear per point, so we use nn.Linear.
# ---------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, encoder_dims: int = 256):
        super().__init__()
        # first_conv: Linear(3->128) + BN + ReLU, Linear(128->256)
        self.fc1 = nn.Linear(3, 128)
        self.bn1 = _BatchNorm(128)
        self.fc2 = nn.Linear(128, 256)
        # second_conv: Linear(512->512) + BN + ReLU, Linear(512->encoder_dims)
        self.fc3 = nn.Linear(512, 512)
        self.bn3 = _BatchNorm(512)
        self.fc4 = nn.Linear(512, encoder_dims)

    def __call__(self, pts: mx.array) -> mx.array:
        """pts: (G, M, 3) -> (G, encoder_dims)"""
        G, M, _ = pts.shape

        x = nn.relu(self.bn1(self.fc1(pts)))          # (G, M, 128)
        x = self.fc2(x)                                # (G, M, 256)

        # global max-pool then broadcast (mirrors feature_global.repeat in PyTorch)
        x_g = x.max(axis=1, keepdims=True)             # (G, 1, 256)
        x_g = mx.broadcast_to(x_g, (G, M, 256))
        x   = mx.concatenate([x_g, x], axis=-1)        # (G, M, 512)

        x = nn.relu(self.bn3(self.fc3(x)))              # (G, M, 512)
        x = self.fc4(x)                                 # (G, M, encoder_dims)
        return x.max(axis=1)                            # (G, encoder_dims)


# ---------------------------------------------------------------------------
# Transformer components
# ---------------------------------------------------------------------------

class _Attention(nn.Module):
    """Multi-head self-attention. qkv_bias=False matches PatchAlign3D default."""

    def __init__(self, dim: int, num_heads: int = 6):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim  = dim // num_heads
        self.scale     = self.head_dim ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=False)
        self.proj      = nn.Linear(dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        H, D    = self.num_heads, self.head_dim

        # (B, N, 3*C) -> (B, N, 3, H, D) -> (B, H, 3, N, D)
        qkv = self.qkv(x).reshape(B, N, 3, H, D).transpose(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # each (B, H, N, D)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale      # (B, H, N, N)
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, N, C)  # (B, N, C)
        return self.proj(out)


class _Block(nn.Module):
    """Transformer block. DropPath is omitted (identity at inference)."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = _Attention(dim, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        hid        = int(dim * mlp_ratio)
        self.fc1   = nn.Linear(dim, hid)
        self.fc2   = nn.Linear(hid, dim)

    def __call__(self, x: mx.array) -> mx.array:
        x = x + self.attn(self.norm1(x))
        h = nn.gelu(self.fc1(self.norm2(x)))
        return x + self.fc2(h)


class _TransformerEncoder(nn.Module):
    """Positional embedding added before every block, matching the original."""

    def __init__(self, embed_dim: int = 384, depth: int = 12, num_heads: int = 6):
        super().__init__()
        self.blocks = [_Block(embed_dim, num_heads) for _ in range(depth)]

    def __call__(self, x: mx.array, pos: mx.array) -> mx.array:
        for blk in self.blocks:
            x = blk(x + pos)
        return x


# ---------------------------------------------------------------------------
# Full model
# ---------------------------------------------------------------------------

class PointTransformerMLX(nn.Module):
    """Inference wrapper: neighborhood + centers -> L2-normalised patch embeddings.

    Usage:
        model = PointTransformerMLX.from_npz("phase3_weights.npz")
        patch_emb = model(neighborhood_mx, centers_mx)  # (G, 1280)
    """

    def __init__(
        self,
        trans_dim: int    = 384,
        depth: int        = 12,
        num_heads: int    = 6,
        encoder_dims: int = 256,
        clip_dim: int     = 1280,
    ):
        super().__init__()
        self.encoder       = _Encoder(encoder_dims)
        self.reduce_dim    = nn.Linear(encoder_dims, trans_dim)
        self.cls_token     = mx.zeros((1, 1, trans_dim))
        self.cls_pos       = mx.zeros((1, 1, trans_dim))
        self.pos_embed_fc1 = nn.Linear(3, 128)
        self.pos_embed_fc2 = nn.Linear(128, trans_dim)
        self.blocks        = _TransformerEncoder(trans_dim, depth, num_heads)
        self.norm          = nn.LayerNorm(trans_dim)
        self.proj          = nn.Linear(trans_dim, clip_dim)

    def __call__(
        self,
        neighborhood: mx.array,   # (G, M, 3) relative coords
        centers: mx.array,         # (G, 3)
    ) -> mx.array:
        """Returns L2-normalised patch embeddings (G, clip_dim)."""
        tok = self.encoder(neighborhood)               # (G, encoder_dims)
        tok = self.reduce_dim(tok)[None]               # (1, G, trans_dim)

        pos = nn.gelu(self.pos_embed_fc1(centers[None]))   # (1, G, 128)
        pos = self.pos_embed_fc2(pos)                       # (1, G, trans_dim)

        x   = mx.concatenate([self.cls_token, tok], axis=1)   # (1, G+1, trans_dim)
        pos = mx.concatenate([self.cls_pos,   pos], axis=1)   # (1, G+1, trans_dim)

        x   = self.blocks(x, pos)                             # (1, G+1, trans_dim)
        x   = self.norm(x)[:, 1:, :][0]                      # (G, trans_dim) drop CLS

        x   = self.proj(x)                                    # (G, clip_dim)
        return x / mx.sqrt((x ** 2).sum(axis=-1, keepdims=True) + 1e-12)

    # ------------------------------------------------------------------
    # Weight loading
    # ------------------------------------------------------------------

    @classmethod
    def from_npz(cls, npz_path: str, **kwargs) -> "PointTransformerMLX":
        npz   = np.load(npz_path)
        model = cls(**kwargs)
        model._load_weights(npz)
        mx.eval(model.parameters())
        return model

    def _load_weights(self, npz) -> None:
        def w(key: str) -> mx.array:
            return mx.array(npz[key].astype(np.float32))

        def load_conv1d(linear: nn.Linear, wkey: str, bkey: str) -> None:
            # PyTorch Conv1d weight (out, in, 1) -> squeeze -> (out, in) == nn.Linear weight
            linear.weight = mx.array(npz[wkey].squeeze(-1).astype(np.float32))
            linear.bias   = mx.array(npz[bkey].astype(np.float32))

        def load_bn(bn: _BatchNorm, prefix: str) -> None:
            bn.weight       = w(f"{prefix}.weight")
            bn.bias         = w(f"{prefix}.bias")
            bn.running_mean = w(f"{prefix}.running_mean")
            bn.running_var  = w(f"{prefix}.running_var")

        ep = "student.encoder"
        load_conv1d(self.encoder.fc1, f"{ep}.first_conv.0.weight",  f"{ep}.first_conv.0.bias")
        load_bn(self.encoder.bn1,                                     f"{ep}.first_conv.1")
        load_conv1d(self.encoder.fc2, f"{ep}.first_conv.3.weight",   f"{ep}.first_conv.3.bias")
        load_conv1d(self.encoder.fc3, f"{ep}.second_conv.0.weight",  f"{ep}.second_conv.0.bias")
        load_bn(self.encoder.bn3,                                     f"{ep}.second_conv.1")
        load_conv1d(self.encoder.fc4, f"{ep}.second_conv.3.weight",  f"{ep}.second_conv.3.bias")

        self.reduce_dim.weight = w("student.reduce_dim.weight")
        self.reduce_dim.bias   = w("student.reduce_dim.bias")

        self.cls_token = w("student.cls_token")
        self.cls_pos   = w("student.cls_pos")

        self.pos_embed_fc1.weight = w("student.pos_embed.0.weight")
        self.pos_embed_fc1.bias   = w("student.pos_embed.0.bias")
        self.pos_embed_fc2.weight = w("student.pos_embed.2.weight")
        self.pos_embed_fc2.bias   = w("student.pos_embed.2.bias")

        for i, blk in enumerate(self.blocks.blocks):
            bp = f"student.blocks.blocks.{i}"
            blk.norm1.weight     = w(f"{bp}.norm1.weight")
            blk.norm1.bias       = w(f"{bp}.norm1.bias")
            blk.norm2.weight     = w(f"{bp}.norm2.weight")
            blk.norm2.bias       = w(f"{bp}.norm2.bias")
            blk.attn.qkv.weight  = w(f"{bp}.attn.qkv.weight")
            blk.attn.proj.weight = w(f"{bp}.attn.proj.weight")
            blk.attn.proj.bias   = w(f"{bp}.attn.proj.bias")
            blk.fc1.weight       = w(f"{bp}.mlp.fc1.weight")
            blk.fc1.bias         = w(f"{bp}.mlp.fc1.bias")
            blk.fc2.weight       = w(f"{bp}.mlp.fc2.weight")
            blk.fc2.bias         = w(f"{bp}.mlp.fc2.bias")

        self.norm.weight = w("student.norm.weight")
        self.norm.bias   = w("student.norm.bias")

        self.proj.weight = w("proj.proj.weight")
        self.proj.bias   = w("proj.proj.bias")
