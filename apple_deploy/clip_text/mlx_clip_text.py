"""MLX CLIP ViT-bigG-14 text encoder — real-time text embedding on Mac.

Architecture (confirmed from open_clip ViT-bigG-14 config):
  vocab_size=49408, context_length=77, width=1280, layers=32, heads=20
  QuickGELU activation: x * sigmoid(1.702 * x)
  Causal attention mask (upper-triangular -inf)
  EOS selection: position of argmax token ID (EOT=49407 is highest)
  Output: (B, 1280) L2-normalised

Weights: clip_text_weights.npz from extract_clip_weights.py
"""
from __future__ import annotations

import numpy as np
import mlx.core as mx
import mlx.nn as nn


def _quick_gelu(x: mx.array) -> mx.array:
    return x * mx.sigmoid(1.702 * x)


class _MLP(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.c_fc   = nn.Linear(width, hidden)
        self.c_proj = nn.Linear(hidden, width)

    def __call__(self, x: mx.array) -> mx.array:
        return self.c_proj(_quick_gelu(self.c_fc(x)))


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads    = heads
        self.head_dim = width // heads
        self.scale    = self.head_dim ** -0.5
        self.in_proj  = nn.Linear(width, width * 3)   # fused QKV, bias=True
        self.out_proj = nn.Linear(width, width)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        B, L, C = x.shape
        H, D = self.heads, self.head_dim

        qkv = self.in_proj(x).reshape(B, L, 3, H, D).transpose(0, 3, 2, 1, 4)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]   # (B, H, L, D)

        attn = (q @ k.transpose(0, 1, 3, 2)) * self.scale      # (B, H, L, L)
        attn = attn + mask                                       # broadcast causal mask (1,1,L,L)
        attn = mx.softmax(attn, axis=-1)

        out = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, C)
        return self.out_proj(out)


class _Block(nn.Module):
    """Pre-LN transformer block matching OpenCLIP ResidualAttentionBlock."""

    def __init__(self, width: int, heads: int, mlp_hidden: int):
        super().__init__()
        self.ln_1 = nn.LayerNorm(width)
        self.attn = _Attention(width, heads)
        self.ln_2 = nn.LayerNorm(width)
        self.mlp  = _MLP(width, mlp_hidden)

    def __call__(self, x: mx.array, mask: mx.array) -> mx.array:
        x = x + self.attn(self.ln_1(x), mask)
        x = x + self.mlp(self.ln_2(x))
        return x


class CLIPTextMLX(nn.Module):
    """CLIP ViT-bigG-14 text encoder in MLX.

    Usage:
        enc = CLIPTextMLX.from_npz("clip_text_weights.npz")
        emb = enc.encode(["left mandibular condyle", "right ramus"])  # (2, 1280)
    """

    VOCAB_SIZE     = 49408
    CONTEXT_LENGTH = 77
    WIDTH          = 1280
    LAYERS         = 32
    HEADS          = 20
    MLP_HIDDEN     = 5120   # WIDTH * 4

    def __init__(self):
        super().__init__()
        W = self.WIDTH
        self.token_embedding      = nn.Embedding(self.VOCAB_SIZE, W)
        self.positional_embedding = mx.zeros((self.CONTEXT_LENGTH, W))
        self.blocks               = [_Block(W, self.HEADS, self.MLP_HIDDEN)
                                      for _ in range(self.LAYERS)]
        self.ln_final             = nn.LayerNorm(W)
        self.text_projection      = mx.zeros((W, W))

        # Causal mask: (L, L) upper-triangular -inf, lower+diag = 0
        mask = np.triu(np.full((self.CONTEXT_LENGTH, self.CONTEXT_LENGTH),
                               -np.inf, dtype=np.float32), k=1)
        self._mask = mx.array(mask)[None, None]   # (1, 1, L, L)

    def __call__(self, token_ids: mx.array) -> mx.array:
        """token_ids: (B, 77) int32  →  (B, 1280) L2-normalised."""
        B = token_ids.shape[0]
        x = self.token_embedding(token_ids) + self.positional_embedding[None]
        for blk in self.blocks:
            x = blk(x, self._mask)
        x = self.ln_final(x)
        # EOS token is the one with the highest ID (49407) in the sequence
        eos_pos = token_ids.argmax(axis=-1)         # (B,)
        x_eos   = x[mx.arange(B), eos_pos]         # (B, W)
        x_eos   = x_eos @ self.text_projection      # (B, W)
        return x_eos / mx.sqrt((x_eos ** 2).sum(axis=-1, keepdims=True) + 1e-12)

    def encode(self, texts: list[str]) -> mx.array:
        """Tokenize and encode texts. Requires open_clip for tokenisation."""
        try:
            import open_clip, torch
            tokenizer = open_clip.get_tokenizer("ViT-bigG-14")
            ids = tokenizer(texts).numpy().astype(np.int32)
        except ImportError:
            raise ImportError("pip install open_clip_torch (needed for tokenisation)")
        emb = self(mx.array(ids))
        mx.eval(emb)
        return emb

    @classmethod
    def from_npz(cls, npz_path: str) -> "CLIPTextMLX":
        npz   = np.load(npz_path)
        model = cls()
        model._load_weights(npz)
        mx.eval(model.parameters())
        return model

    def _load_weights(self, npz) -> None:
        def w(key: str) -> mx.array:
            return mx.array(npz[key].astype(np.float32))

        self.token_embedding.weight   = w("token_embedding.weight")
        self.positional_embedding     = w("positional_embedding")
        self.ln_final.weight          = w("ln_final.weight")
        self.ln_final.bias            = w("ln_final.bias")
        self.text_projection          = w("text_projection")

        for i, blk in enumerate(self.blocks):
            p = f"transformer.resblocks.{i}"
            blk.ln_1.weight          = w(f"{p}.ln_1.weight")
            blk.ln_1.bias            = w(f"{p}.ln_1.bias")
            blk.ln_2.weight          = w(f"{p}.ln_2.weight")
            blk.ln_2.bias            = w(f"{p}.ln_2.bias")
            blk.attn.in_proj.weight  = w(f"{p}.attn.in_proj_weight")
            if f"{p}.attn.in_proj_bias" in npz.files:
                blk.attn.in_proj.bias = w(f"{p}.attn.in_proj_bias")
            blk.attn.out_proj.weight = w(f"{p}.attn.out_proj.weight")
            blk.attn.out_proj.bias   = w(f"{p}.attn.out_proj.bias")
            blk.mlp.c_fc.weight      = w(f"{p}.mlp.c_fc.weight")
            blk.mlp.c_fc.bias        = w(f"{p}.mlp.c_fc.bias")
            blk.mlp.c_proj.weight    = w(f"{p}.mlp.c_proj.weight")
            blk.mlp.c_proj.bias      = w(f"{p}.mlp.c_proj.bias")
