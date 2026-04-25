# ViT-bigG-14 テキストエンコーダ アーキテクチャ

## 概要

| 項目 | 値 |
|------|-----|
| モデル名 | `ViT-bigG-14` / pretrained `laion2b_s39b_b160k` |
| テキストバックボーン | CLIPTextTransformer (GPT-2 スタイル Transformer) |
| hidden size | 1280 |
| Transformer 層数 | 32 |
| attention heads | 20 |
| intermediate size (FFN) | 5120 |
| vocabulary size | 49,408 |
| 最大トークン長 | 77 |
| テキスト出力次元 | **1280** (projection 後) |
| テキスト部 総パラメータ数 | ~401M |

BiomedCLIP との最大の違いは、モデルクラスが `CLIP`（標準）であること。
テキストモジュールが `clip_model.text`（HFTextEncoder）ではなく、
`clip_model.transformer`（CLIPTextTransformer）に格納される。

---

## open_clip でのロード後の構造

`open_clip.create_model_and_transforms("ViT-bigG-14", pretrained="laion2b_s39b_b160k")` で
返されるモデルは標準 `CLIP` クラス。

```
clip_model  (CLIP)
├── visual           : ViT-bigG/14 画像エンコーダ（このプロジェクトでは使用しない）
├── transformer      : CLIPTextTransformer
│   └── resblocks   : ModuleList (32 層)
│       └── [0..31] : ResidualAttentionBlock
│           ├── attn      : MultiheadAttention
│           │   ├── in_proj_weight  Linear(1280, 3840)  [Q/K/V 結合]
│           │   ├── in_proj_bias    (3840,)
│           │   ├── out_proj        Linear(1280, 1280)
│           │   └── out_proj.bias   (1280,)
│           ├── ln_1      : LayerNorm(1280)
│           ├── mlp       : Sequential
│           │   ├── [0] c_fc    Linear(1280, 5120)  [GELU]
│           │   └── [2] c_proj  Linear(5120, 1280)
│           └── ln_2      : LayerNorm(1280)
├── token_embedding  : Embedding(49408, 1280)
├── positional_embedding : Parameter(77, 1280)
├── ln_final         : LayerNorm(1280)    ← 解凍対象
└── text_projection  : Parameter(1280, 1280)  ← 解凍対象 (projection)
```

---

## BiomedCLIP との構造比較

| 項目 | BiomedCLIP | ViT-bigG-14 |
|------|------------|-------------|
| モデルクラス | `CustomTextCLIP` | `CLIP`（標準） |
| テキストモジュール | `clip_model.text` (HFTextEncoder) | `clip_model.transformer` (CLIPTextTransformer) |
| Transformer ブロック | `clip_model.text.transformer.encoder.layer` (BertLayer×12) | `clip_model.transformer.resblocks` (ResidualAttentionBlock×32) |
| Projection | `clip_model.text.proj` (Linear 768→512) | `clip_model.text_projection` (Parameter 1280×1280) |
| LayerNorm (最終) | BERT 内部 | `clip_model.ln_final` (独立モジュール) |
| 出力次元 | 512 | **1280** |

---

## 各層のパラメータ数

| コンポーネント | パラメータ数 |
|----------------|-------------|
| token_embedding | ~63.2M |
| positional_embedding | ~98K |
| ResidualAttentionBlock × 1 | ~10.5M |
| ResidualAttentionBlock × 32 | ~336M |
| ln_final | ~2.6K |
| text_projection | ~1.6M |
| **text 部合計** | **~401M** |

---

## Stage 3b での解凍対象

`freeze_text_encoder_except_last(clip_model, n_last=1)` が以下を解凍:

| 対象 | アクセスパス | パラメータ数 |
|------|-------------|-------------|
| 最終 ResidualAttentionBlock (resblocks[31]) | `clip_model.transformer.resblocks[-1]` | ~10.5M |
| 最終 LayerNorm | `clip_model.ln_final` | ~2.6K |
| テキスト投影 | `clip_model.text_projection` | ~1.6M |
| **合計** | | **~12.1M** (text 部全体の約 3.0%) |

### 解凍する理由

- **最終 ResidualAttentionBlock**: タスク特化の文脈表現を調整。Q/K/V attention 重みと FFN が含まれるため、テキスト間の識別力を直接向上できる。
- **ln_final**: 最終 Transformer 出力を正規化する層。最終ブロックを解凍する場合はこれも合わせて解凍しないと勾配が途中で止まる。
- **text_projection (Parameter)**: 埋め込み空間への投影を微調整することで、コサイン類似度の分布を改善できる。
- **中間層・token_embedding**: 凍結のまま。LAION-2B の汎用ドメイン知識を保持するため。

### Stage 3a では凍結のまま

`train_stage3a_bigG14.py` では `text_tune` を使わない。
大まかな方向の位置合わせ段階では、テキスト埋め込み空間は固定して point encoder だけを学習する。

---

## 学習時の gradient の流れ

```
テキスト文字列
    ↓ tokenizer
tokens (B, L)
    ↓ clip_model.encode_text()  ← encode_with_grad() 経由で呼ぶ
        resblocks[0..30]   (凍結: grad なし)
        resblocks[31]      (解凍: grad あり) ← ここから backward が流れる
        ln_final           (解凍: grad あり)
        text_projection    (解凍: grad あり)
    ↓
text_feats (K, 1280)  正規化済み
    ↓
dental_bce_loss / infonce_loss
    ↓
.backward()
```

- `encode_with_grad()` はキャッシュをバイパスして毎回 `clip_model.encode_text` を呼ぶ
- `encode_raw_texts()` は `@torch.no_grad()` → 評価・teacher パスのみ使用

---

## Learning Rate 設定

| コンポーネント | LR | 備考 |
|----------------|-----|------|
| Point encoder (student) | `3e-5` | |
| PatchToTextProj | `3e-5` | |
| LearnableTemp | `3e-5` | |
| **Text encoder 最終層 + proj** | **`1e-5`** | 事前知識の破壊を防ぐため小さく |

テキストエンコーダは LAION-2B 大規模事前学習済みのため、point encoder と同じ LR では破滅的忘却が起きやすい。

---

## チェックポイントのキー構造

### 学習中の `last.pt` / `best.pt`

```python
{
    "epoch":         int,
    "student":       OrderedDict,   # point encoder (point_transformer)
    "proj":          OrderedDict,   # PatchToTextProj (Linear 384→1280)
    "temp":          OrderedDict,   # LearnableTemp
    "text_encoder":  {              # ViT-bigG-14 テキスト部  ← Stage 3b 追加
        "transformer":      OrderedDict,  # clip_model.transformer.state_dict()
        "ln_final":         OrderedDict,  # clip_model.ln_final.state_dict()
        "text_projection":  Tensor,       # clip_model.text_projection.data
    },
    "optimizer":     ...,
    "scheduler":     ...,
    "best_val_loss": float,
    "train_metrics": dict,
    "val_metrics":   dict,
}
```

### 推論用 `stage3b_best.pt`

```python
{
    "epoch":        int,
    "student":      OrderedDict,   # point encoder
    "proj":         OrderedDict,   # PatchToTextProj
    "text_encoder": {              # ViT-bigG-14 テキスト部  ← Stage 3b 追加
        "transformer":     OrderedDict,
        "ln_final":        OrderedDict,
        "text_projection": Tensor,
    },
}
```

---

## 推論時のロード例

```python
import open_clip
import torch

clip_model, _, _ = open_clip.create_model_and_transforms(
    "ViT-bigG-14", pretrained="laion2b_s39b_b160k"
)
tokenizer = open_clip.get_tokenizer("ViT-bigG-14")

ckpt = torch.load("stage3b_best.pt", map_location="cpu")
if "text_encoder" in ckpt:
    te = ckpt["text_encoder"]
    clip_model.transformer.load_state_dict(te["transformer"], strict=False)
    if "ln_final" in te:
        clip_model.ln_final.load_state_dict(te["ln_final"], strict=False)
    if "text_projection" in te and te["text_projection"] is not None:
        clip_model.text_projection.data.copy_(te["text_projection"])

clip_model.eval()
```
