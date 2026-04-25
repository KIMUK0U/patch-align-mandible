# BiomedCLIP テキストエンコーダ アーキテクチャ

## 概要

| 項目 | 値 |
|------|-----|
| モデル名 | `hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224` |
| テキストバックボーン | PubMedBERT (BERT-base 構造, 生医学テキストで事前学習) |
| hidden size | 768 |
| Transformer 層数 | 12 |
| attention heads | 12 |
| intermediate size (FFN) | 3072 |
| vocabulary size | 28,895 (BioMed 特化) |
| 最大トークン長 | 256 |
| テキスト出力次元 | **512** (projection 後) |
| テキスト部 総パラメータ数 | ~86M |

---

## open_clip でのロード後の構造

`open_clip.create_model_and_transforms("hf-hub:microsoft/BiomedCLIP-...")` で返される
モデルは `CustomTextCLIP` クラス。テキスト部は `HFTextEncoder` ラッパー。

```
clip_model  (CustomTextCLIP)
├── visual      : ViT-B/16 画像エンコーダ（このプロジェクトでは使用しない）
└── text        : HFTextEncoder
    ├── transformer : BertModel  (PubMedBERT)
    │   ├── embeddings  : BertEmbeddings
    │   │   ├── word_embeddings       Embedding(28895, 768)
    │   │   ├── position_embeddings   Embedding(512, 768)
    │   │   ├── token_type_embeddings Embedding(2, 768)
    │   │   ├── LayerNorm(768)
    │   │   └── dropout
    │   └── encoder : BertEncoder
    │       └── layer : ModuleList (12 層)
    │           └── [0..11] : BertLayer
    │               ├── attention : BertAttention
    │               │   ├── self : BertSelfAttention
    │               │   │   ├── query  Linear(768, 768)
    │               │   │   ├── key    Linear(768, 768)
    │               │   │   ├── value  Linear(768, 768)
    │               │   │   └── dropout
    │               │   └── output : BertSelfOutput
    │               │       ├── dense    Linear(768, 768)
    │               │       ├── LayerNorm(768)
    │               │       └── dropout
    │               ├── intermediate : BertIntermediate
    │               │   └── dense    Linear(768, 3072)  [GELU]
    │               └── output : BertOutput
    │                   ├── dense    Linear(3072, 768)
    │                   ├── LayerNorm(768)
    │                   └── dropout
    └── proj : Linear(768, 512, bias=False)   ← テキスト投影層
```

---

## 各層のパラメータ数

| コンポーネント | パラメータ数 |
|----------------|-------------|
| embeddings | ~22.7M |
| BertLayer × 1 | ~7.1M |
| BertLayer × 12 (全層) | ~85.1M |
| proj Linear(768→512) | ~393K |
| **text 部合計** | **~108M** |

---

## Stage 3b での解凍対象

`freeze_text_encoder_except_last(clip_model, n_last=1)` が以下を解凍:

| 対象 | アクセスパス | パラメータ数 |
|------|-------------|-------------|
| 最終 BertLayer (layer[11]) | `clip_model.text.transformer.encoder.layer[-1]` | ~7.1M |
| テキスト投影層 | `clip_model.text.proj` | ~393K |
| **合計** | | **~7.5M** (text 部全体の約 8.7%) |

### 解凍する理由

- **最終 Transformer 層**: タスク特化の文脈表現を調整。Q/K/V attention 重みと FFN が含まれるため、テキスト間の識別力を直接向上できる。
- **projection 層 (768→512)**: 埋め込み空間への投影を微調整することで、コサイン類似度の分布を改善できる。
- **中間層・embedding 層**: 凍結のまま。PubMedBERT の biomedical domain 事前知識を保持するため。

### Stage 3a では凍結のまま

`train_stage3a.py`（および `phase3_training.ipynb`）では `text_tune.enabled: false` 相当の動作をする。
大まかな方向の位置合わせ段階では、テキスト埋め込み空間は固定して point encoder だけを学習する。

---

## 学習時の gradient の流れ

```
テキスト文字列
    ↓ tokenizer
tokens (B, L)
    ↓ clip_model.encode_text()  ← encode_with_grad() 経由で呼ぶ
        BertLayer[0..10]  (凍結: grad なし)
        BertLayer[11]     (解凍: grad あり) ← ここから backward が流れる
        proj Linear       (解凍: grad あり)
    ↓
text_feats (K, 512)  正規化済み
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

テキストエンコーダは大規模事前学習済みのため、point encoder と同じ LR では破滅的忘却が起きやすい。

---

## チェックポイントのキー構造

### 学習中の `last.pt` / `best.pt`

```python
{
    "epoch":         int,
    "student":       OrderedDict,   # point encoder (point_transformer)
    "proj":          OrderedDict,   # PatchToTextProj (Linear 384→512)
    "temp":          OrderedDict,   # LearnableTemp
    "text_encoder":  OrderedDict,   # clip_model.text (HFTextEncoder 全体)  ← Stage 3b 追加
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
    "text_encoder": OrderedDict,   # clip_model.text  ← Stage 3b 追加
}
```

---

## 推論時のロード例

```python
import open_clip
import torch

MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

clip_model, _, _ = open_clip.create_model_and_transforms(MODEL_NAME)
tokenizer = open_clip.get_tokenizer(MODEL_NAME)

ckpt = torch.load("stage3b_best.pt", map_location="cpu")
if "text_encoder" in ckpt:
    clip_model.text.load_state_dict(ckpt["text_encoder"], strict=False)

clip_model.eval()
```
