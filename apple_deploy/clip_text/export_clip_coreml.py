"""Export CLIP ViT-bigG-14 text encoder → CoreML .mlpackage for iOS/visionOS.

Input:  int32[1, 77]  — BPE token IDs, padded/truncated to 77
Output: float32[1, 1280] — L2-normalised text embedding

Requirements:
    pip install open_clip_torch torch coremltools

Usage:
    cd Phase3_MLX
    python clip_text/export_clip_coreml.py
    python clip_text/export_clip_coreml.py --out dental_clip_text.mlpackage
"""
import argparse
import numpy as np
import torch
import torch.nn as nn
import open_clip
import coremltools as ct


class CLIPTextWrapper(nn.Module):
    """Wraps OpenCLIP ViT-bigG-14 text encoder for CoreML tracing (batch=1).

    Forces float32; strips visual encoder.
    Fixed causal mask avoids recomputation; EOS extracted via argmax.
    """

    def __init__(self, clip_model: nn.Module):
        super().__init__()
        clip_model = clip_model.float()
        self.token_embedding      = clip_model.token_embedding
        self.positional_embedding = nn.Parameter(clip_model.positional_embedding)
        self.transformer          = clip_model.transformer
        self.ln_final             = clip_model.ln_final
        self.text_projection      = nn.Parameter(clip_model.text_projection)

        L = 77
        mask = torch.triu(torch.full((L, L), float("-inf")), diagonal=1)
        self.register_buffer("attn_mask", mask)   # (77, 77)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        """token_ids: int32 (1, 77) → (1, 1280) float32 L2-normalised."""
        x = self.token_embedding(token_ids).float() + self.positional_embedding  # (1, 77, C)
        # open_clip's Transformer handles NLD/LND format internally.
        x = self.transformer(x, attn_mask=self.attn_mask)                        # (1, 77, C)
        x = self.ln_final(x)

        # EOS is the token with the highest ID (49407) in the sequence
        eos = token_ids[0].argmax()   # scalar tensor — traceable
        x_eos = x[0, eos]            # (C,)
        x_eos = x_eos @ self.text_projection                                     # (C,)
        x_eos = x_eos / (x_eos.norm(dim=-1, keepdim=True) + 1e-12)
        return x_eos.unsqueeze(0)    # (1, 1280)


def _make_sample_ids() -> torch.Tensor:
    """Sample token IDs: SOT + placeholder + EOT, padded to 77."""
    ids = torch.zeros(1, 77, dtype=torch.int32)
    ids[0, 0] = 49406   # <|startoftext|>
    ids[0, 1] = 21588   # placeholder token
    ids[0, 2] = 49407   # <|endoftext|> (highest ID → argmax finds it)
    return ids


def export(model_name: str, pretrained: str, out_path: str) -> None:
    print(f"Loading {model_name} / {pretrained} ...")
    clip_model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    clip_model.eval()

    wrapper = CLIPTextWrapper(clip_model).eval()

    sample_ids = _make_sample_ids()

    with torch.no_grad():
        out = wrapper(sample_ids)
    assert out.shape == (1, 1280), f"Unexpected shape: {out.shape}"
    print(f"  Forward pass OK — output shape: {out.shape}")

    print("Tracing model (torch.jit.trace) ...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (sample_ids,))

    print("Converting to CoreML ...")
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="token_ids", shape=(1, 77), dtype=np.int32)],
        outputs=[ct.TensorType(name="text_embedding", dtype=np.float32)],
        minimum_deployment_target=ct.target.iOS17,
        compute_precision=ct.precision.FLOAT32,
    )

    out_path = out_path if out_path.endswith(".mlpackage") else out_path + ".mlpackage"
    mlmodel.save(out_path)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="ViT-bigG-14")
    parser.add_argument("--pretrained", default="laion2b_s39b_b160k")
    parser.add_argument("--out",        default="dental_clip_text.mlpackage")
    args = parser.parse_args()
    export(args.model, args.pretrained, args.out)
