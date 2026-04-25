"""Export Phase3 model to Core ML for iOS / visionOS deployment.

The exported model takes PRE-COMPUTED patch neighborhoods as input.
FPS + KNN preprocessing is implemented in Swift (DentalInferenceKit/).

Architecture exported:
    Input  : neighborhood float32[G, M, 3]  (relative coords, from Swift FPS+KNN)
             centers     float32[G, 3]
    Output : patch_embeddings float32[G, clip_dim]  (L2-normalised)

Default: G=128, M=32, clip_dim=1280, minimum_deployment_target=iOS17

Requirements:
    pip install coremltools torch

Usage:
    python export_coreml.py --ckpt ../Phase3_Training/outputs/stage3a/stage3a_last.pt
    python export_coreml.py --ckpt best.pt --out MyModel.mlpackage --G 128 --M 32
"""
import argparse
import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------
# Traceable PyTorch model (no pointnet2_ops / knn_cuda)
# Takes pre-grouped neighborhoods; FPS+KNN is done in Swift.
# --------------------------------------------------------------------------

class _Encoder(nn.Module):
    def __init__(self, encoder_dims: int = 256):
        super().__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_dims, 1),
        )

    def forward(self, neighborhood: torch.Tensor) -> torch.Tensor:
        # neighborhood: (G, M, 3)
        G, M, _ = neighborhood.shape
        x    = neighborhood.permute(0, 2, 1)                          # (G, 3, M)
        feat = self.first_conv(x)                                      # (G, 256, M)
        feat_g = feat.max(dim=2, keepdim=True)[0].expand(-1, -1, M)
        feat = torch.cat([feat_g, feat], dim=1)                        # (G, 512, M)
        feat = self.second_conv(feat)                                   # (G, enc_dims, M)
        return feat.max(dim=2)[0]                                       # (G, enc_dims)


class DentalPatchNet(nn.Module):
    """Traceable inference model for Core ML export.

    Uses nn.TransformerEncoder (standard PyTorch) instead of the custom
    TransformerEncoder from PatchAlign3D so that torch.jit.trace works without
    CUDA-only ops.  Weights are remapped in load_phase3_weights().
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
        self.encoder    = _Encoder(encoder_dims)
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos    = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.pos_embed  = nn.Sequential(
            nn.Linear(3, 128), nn.GELU(), nn.Linear(128, trans_dim)
        )
        enc_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim, nhead=num_heads,
            dim_feedforward=trans_dim * 4,
            dropout=0.0, batch_first=True,
            activation="gelu", norm_first=False,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=depth)
        self.norm        = nn.LayerNorm(trans_dim)
        self.proj        = nn.Linear(trans_dim, clip_dim)

    def forward(
        self,
        neighborhood: torch.Tensor,   # (G, M, 3)
        centers: torch.Tensor,         # (G, 3)
    ) -> torch.Tensor:
        G = neighborhood.shape[0]
        tok = self.reduce_dim(self.encoder(neighborhood)).unsqueeze(0)  # (1, G, D)
        pos = self.pos_embed(centers).unsqueeze(0)                      # (1, G, D)

        x   = torch.cat([self.cls_token, tok], dim=1)                  # (1, G+1, D)
        pos = torch.cat([self.cls_pos,   pos], dim=1)                  # (1, G+1, D)
        x   = self.transformer(x + pos)                                 # (1, G+1, D)
        x   = self.norm(x)[:, 1:, :].squeeze(0)                        # (G, D)
        return F.normalize(self.proj(x), dim=-1)                        # (G, clip_dim)


# --------------------------------------------------------------------------
# Weight loading
# --------------------------------------------------------------------------

def _strip_module(state: dict) -> dict:
    return OrderedDict((k.replace("module.", ""), v) for k, v in state.items())


def load_phase3_weights(model: DentalPatchNet, ckpt_path: str) -> None:
    ckpt    = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    student = _strip_module(ckpt.get("student", {}))
    proj    = _strip_module(ckpt.get("proj", {}))

    mapped: dict = {}

    # Direct copy: encoder, reduce_dim, cls_token/pos, pos_embed, norm
    for k, v in student.items():
        for prefix in ("encoder.", "reduce_dim.", "cls_token", "cls_pos", "pos_embed.", "norm."):
            if k.startswith(prefix) or k == prefix.rstrip("."):
                mapped[k] = v

    # Transformer blocks: PatchAlign3D -> nn.TransformerEncoderLayer
    for i in range(12):
        sp = f"blocks.blocks.{i}"
        tp = f"transformer.layers.{i}"
        remap = [
            (f"{sp}.norm1.weight",     f"{tp}.norm1.weight"),
            (f"{sp}.norm1.bias",       f"{tp}.norm1.bias"),
            (f"{sp}.norm2.weight",     f"{tp}.norm2.weight"),
            (f"{sp}.norm2.bias",       f"{tp}.norm2.bias"),
            (f"{sp}.attn.qkv.weight",  f"{tp}.self_attn.in_proj_weight"),
            (f"{sp}.attn.proj.weight", f"{tp}.self_attn.out_proj.weight"),
            (f"{sp}.attn.proj.bias",   f"{tp}.self_attn.out_proj.bias"),
            (f"{sp}.mlp.fc1.weight",   f"{tp}.linear1.weight"),
            (f"{sp}.mlp.fc1.bias",     f"{tp}.linear1.bias"),
            (f"{sp}.mlp.fc2.weight",   f"{tp}.linear2.weight"),
            (f"{sp}.mlp.fc2.bias",     f"{tp}.linear2.bias"),
        ]
        for src, dst in remap:
            if src in student:
                mapped[dst] = student[src]
        # PatchAlign3D uses qkv_bias=False; nn.TransformerEncoderLayer needs a bias tensor
        in_proj_w = mapped.get(f"{tp}.self_attn.in_proj_weight")
        if in_proj_w is not None:
            mapped[f"{tp}.self_attn.in_proj_bias"] = torch.zeros(in_proj_w.shape[0])

    mapped["proj.weight"] = proj.get("proj.weight", torch.zeros(1280, 384))
    mapped["proj.bias"]   = proj.get("proj.bias",   torch.zeros(1280))

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    if missing:
        print(f"[warn] {len(missing)} missing keys (first 5): {missing[:5]}")
    print(f"Loaded weights from {ckpt_path}")


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------

def export(ckpt_path: str, out_path: str, G: int = 128, M: int = 32) -> None:
    try:
        import coremltools as ct
    except ImportError:
        raise ImportError("pip install coremltools")

    model = DentalPatchNet()
    load_phase3_weights(model, ckpt_path)
    model.eval()

    dummy_neigh   = torch.randn(G, M, 3)
    dummy_centers = torch.randn(G, 3)

    with torch.no_grad():
        traced = torch.jit.trace(model, (dummy_neigh, dummy_centers))

    cml = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="neighborhood", shape=(G, M, 3)),
            ct.TensorType(name="centers",      shape=(G, 3)),
        ],
        outputs=[ct.TensorType(name="patch_embeddings")],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )
    cml.short_description = "DentalPatch3D — per-patch anatomical embeddings"
    cml.save(out_path)
    print(f"Saved Core ML model → {out_path}")
    print(f"  Input  : neighborhood ({G}, {M}, 3)  centers ({G}, 3)")
    print(f"  Output : patch_embeddings ({G}, 1280)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--out",  default="dental_patch_encoder.mlpackage")
    parser.add_argument("--G",    type=int, default=128)
    parser.add_argument("--M",    type=int, default=32)
    args = parser.parse_args()
    export(args.ckpt, args.out, args.G, args.M)
