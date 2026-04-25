"""DentalTextCache: 歯科テキスト専用の CLIP テキストエンコーダキャッシュ。

stage2.py の LRUTextCache を歯科用途に特化して改変。
歯科アノテーションテキストは完全な文 (long-form) なので、
テンプレート展開なしで直接エンコードする。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from collections import OrderedDict


class DentalTextCache:
    """open_clip CLIP モデルを使った歯科テキスト LRU キャッシュ。

    Parameters
    ----------
    clip_model  : open_clip テキストエンコーダを持つモデル
    tokenizer   : open_clip tokenizer
    device      : torch.device
    capacity    : LRU キャッシュ容量
    """

    def __init__(
        self,
        clip_model,
        tokenizer,
        device: torch.device,
        capacity: int = 10000,
    ):
        self.clip_model = clip_model
        self.tokenizer  = tokenizer
        self.device     = device
        self.capacity   = max(512, int(capacity))
        self._store: OrderedDict[str, torch.Tensor] = OrderedDict()

    def _touch(self, key: str) -> None:
        self._store.move_to_end(key)

    @torch.no_grad()
    def _encode_single(self, text: str) -> torch.Tensor:
        """1 テキストを正規化済み 1D Tensor (D,) に変換。"""
        key = text.strip()
        if key in self._store:
            self._touch(key)
            return self._store[key]

        toks = self.tokenizer([key]).to(self.device)
        feat = self.clip_model.encode_text(toks)                   # (1, D)
        feat = F.normalize(feat, dim=-1).squeeze(0).float().cpu()  # (D,)

        self._store[key] = feat
        if len(self._store) > self.capacity:
            self._store.popitem(last=False)

        return feat

    @torch.no_grad()
    def encode_raw_texts(self, texts: list[str]) -> torch.Tensor:
        """テキストのリストを (K, D) Tensor にエンコードする（キャッシュ使用、grad なし）。

        評価・teacher パス・stage3a で使用する。

        Args:
            texts: K 個のテキスト文字列

        Returns:
            (K, D) float32 Tensor (正規化済み)
        """
        feats = [self._encode_single(t) for t in texts]
        if not feats:
            return torch.empty(0, dtype=torch.float32)
        return torch.stack(feats, dim=0)   # (K, D)

    def encode_with_grad(self, texts: list[str]) -> torch.Tensor:
        """テキストのリストを (K, D) Tensor にエンコードする（キャッシュバイパス、grad あり）。

        Stage 3b でテキストエンコーダを学習するときに使用する。
        キャッシュを経由せず毎回 clip_model.encode_text を呼ぶため、
        テキストエンコーダの重みに gradient が流れる。

        Args:
            texts: K 個のテキスト文字列

        Returns:
            (K, D) float32 Tensor (正規化済み)
        """
        if not texts:
            return torch.empty(0, dtype=torch.float32, device=self.device)
        toks = self.tokenizer(texts).to(self.device)
        feats = self.clip_model.encode_text(toks)   # (K, D)
        return F.normalize(feats, dim=-1).float()   # (K, D)


def build_clip_and_cache(
    clip_model_name: str,
    clip_pretrained: str,
    device: torch.device,
    capacity: int = 10000,
) -> tuple[object, DentalTextCache]:
    """open_clip モデルを構築し DentalTextCache を返す。

    Parameters
    ----------
    clip_model_name : 例 "ViT-bigG-14" または
                      "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    clip_pretrained : 例 "laion2b_s39b_b160k"。hf-hub: 形式の場合は空文字でよい。
    device          : torch.device
    capacity        : キャッシュ容量

    Returns
    -------
    (clip_model, text_cache)
    """
    import open_clip  # type: ignore

    # hf-hub: 形式 (BiomedCLIP 等) は pretrained 引数が不要
    is_hf_hub = clip_model_name.startswith("hf-hub:")
    if is_hf_hub:
        clip_model, _, _ = open_clip.create_model_and_transforms(clip_model_name)
    else:
        clip_model, _, _ = open_clip.create_model_and_transforms(
            clip_model_name, pretrained=clip_pretrained
        )
    clip_model = clip_model.to(device)
    clip_model.eval()
    for p in clip_model.parameters():
        p.requires_grad = False

    tokenizer = open_clip.get_tokenizer(clip_model_name)
    text_cache = DentalTextCache(clip_model, tokenizer, device, capacity)

    # テキスト出力次元を検出
    if hasattr(clip_model, "text_projection"):
        text_dim = clip_model.text_projection.shape[-1]
    elif hasattr(clip_model, "text") and hasattr(clip_model.text, "output_dim"):
        text_dim = clip_model.text.output_dim
    else:
        text_dim = "N/A"

    label = clip_model_name if is_hf_hub else f"{clip_model_name}/{clip_pretrained}"
    print(f"[TextCache] CLIP {label} をロード  テキスト次元: {text_dim}")
    return clip_model, text_cache
