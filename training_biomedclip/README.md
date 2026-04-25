# Phase 3: PatchAligned3D 歯科ドメイン適応学習

設計書: `design_docs/PatchAligned3D_phases/phase3_training_plan.md`

---

## ディレクトリ構成

```
Phase3_Training/
├── datasets/
│   └── dental_dataset.py      # DentalPatchDataset (PLY + candidates.json)
├── models/
│   ├── stage3_model.py        # Student / Teacher / PatchToTextProj
│   ├── text_cache.py          # DentalTextCache (CLIP テキストエンコーダ)
│   └── ewc.py                 # EWC クラス + Fisher 推定
├── losses/
│   ├── infonce.py             # Stage 3a: InfoNCE 損失
│   ├── bce_multilabel.py      # Stage 3b: BCE multi-label 損失
│   └── distillation.py        # KD 損失 (MSE + KL)
├── train_stage3a.py           # Stage 3a 実行スクリプト
├── train_stage3b.py           # Stage 3b 実行スクリプト
└── configs/
    ├── stage3a.yaml
    └── stage3b.yaml
```

---

## データ構成

| ファイル | 内容 |
|--------|------|
| `Dataset/6167726/STLs/PLYs/Pat Xa_sample.ply` | 8192 点の点群 (FPS で 2048 点にダウンサンプリング) |
| `data/DentalPatchData/dental_vocab/candidates.json` | 患者ごとに 32 アノテーションパッチ × 5 テキスト |

20 患者 × 32 パッチ = 640 パッチ、3060 ユニークテキスト。

---

## 実行手順

### 事前準備

```bash
cd DentalPatchAligned3D/Phase3_Training

# 依存パッケージ
pip install plyfile open3d open-clip-torch easydict pyyaml tqdm
```

`configs/stage3a.yaml` の `stage2_ckpt` に Stage 2 チェックポイントのパスを設定してください。

---

### Stage 3a 実行 (領域パッチ InfoNCE、30 エポック)

```bash
python train_stage3a.py --config configs/stage3a.yaml
```

出力:
- `outputs/stage3a/checkpoints/best.pt` — 最良チェックポイント
- `outputs/stage3a/stage3a_last.pt` — Stage 3b 入力用

---

### Stage 3b 実行 (局所パッチ BCE + KD + EWC、50 エポック)

```bash
python train_stage3b.py \
    --config configs/stage3b.yaml \
    --stage3a_ckpt outputs/stage3a/stage3a_last.pt
```

出力:
- `outputs/stage3b/checkpoints/best.pt`
- `outputs/stage3b/stage3b_best.pt` — Phase 4 (Attribution) に渡す
- `outputs/stage3b/ewc_fisher.pt` — Fisher 情報行列 (Phase 4 参考用)

---

## 主要設計メモ

### 座標系の扱い
- PLY は元の座標系 (mm スケール)、`candidates.json` の `patch_center_lps` は同じ座標系
- 両者に **同一の単位球正規化** (PLY の重心・最大半径で正規化) を適用
- 正規化後に最近傍アサインメントでラベルマスクを生成

### Stage 3a (octant 粗粒度)
- 9 方位 (L/R/M × A/P × I/S) を領域 ID として InfoNCE 学習
- ラベル: `"left anterior superior region including the condylar process"` など

### Stage 3b (パッチ細粒度)
- 32 アノテーションパッチ × 5 テキストを直接 BCE 学習
- `text_augment=True`: 各イテレーションで 5 テキストからランダム選択 (戦略 A)
- EWC で Stage 3a 重みを保護

### candidates.json (未確認版) の扱い
- 現状は人手確認前の API 生成データを使用
- `texts[0]` が最も信頼性が高い場合は `text_augment=False` に変更
- 確認済みデータが揃い次第、同じスクリプトで再学習可能
