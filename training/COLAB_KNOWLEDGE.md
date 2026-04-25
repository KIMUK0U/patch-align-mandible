# Phase 3 Google Colab 知見まとめ

次プロジェクト向けの環境・データ・重みのリファレンス。

---

## データパス (Google Drive)

| 用途 | パス |
|------|------|
| 点群 PLY (20症例、2048点、正規化済み) | `MyDrive/2026研究/Dataset/6167726/STLs/STLs/normPLYs/` |
| アノテーション JSON | `MyDrive/2026研究/ULIP_PointLLM/data/DentalPatchData/dental_vocab/candidates.json` |
| PLY ファイル命名規則 | `Pat Xa_norm.ply` (例: `Pat 1a_norm.ply`, `Pat 10na_norm.ply`) |
| JSON キー命名規則 | PLY のステム名と一致 (例: `"Pat 1a_norm"`) |

```
candidates.json 構造:
{
  "Pat 1a_norm": [
    {
      "patch_id": 0,
      "patch_center_lps": [x, y, z],
      "lps_octant": "R-A-I",
      "texts": ["text1", "text2", "text3", "text4", "text5"]
    },
    ...  // 32 パッチ
  ],
  ...  // 20 症例
}
```

---

## 学習済み重みパス

| モデル | パス | 備考 |
|--------|------|------|
| PatchAlign3D 公式 Stage 2 | `MyDrive/2026研究/ULIP_PointLLM/checkpoints/patchalign3d/patchalign3d.pt` | HuggingFace `patchalign3d/patchalign3d-encoder` からダウンロード |
| Stage 3a best | `MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training/outputs/stage3a/checkpoints/best.pt` | octant 粗粒度 InfoNCE |
| Stage 3a last | `MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training/outputs/stage3a/stage3a_last.pt` | Stage 3b 入力用 |
| Stage 3b best (Phase4入力) | `MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training/outputs/stage3b/stage3b_best.pt` | パッチ詳細 BCE |
| Stage 3b last | `MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training/outputs/stage3b/checkpoints/last.pt` | |
| EWC Fisher 行列 | `MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training/outputs/stage3b/ewc_fisher.pt` | Stage 3b の EWC 正則化用 |

### チェックポイントのキー構造
```python
ckpt = torch.load(path, map_location='cpu', weights_only=False)
# ckpt.keys(): ['epoch', 'student', 'proj', 'temp', 'optimizer', 'scheduler', 'best_val_loss', 'train_metrics', 'val_metrics']
# stage3b_best.pt: ['student', 'proj', 'epoch']

model.load_state_dict(ckpt['student'], strict=False)
proj.load_state_dict(ckpt['proj'], strict=False)
```

---

## モデル設定 (固定値)

```python
cfg = {
    'trans_dim':      384,
    'depth':          12,
    'drop_path_rate': 0.1,
    'cls_dim':        50,
    'num_heads':      6,
    'group_size':     32,   # 1パッチあたりの点数 M
    'num_group':      128,  # パッチ数 G
    'encoder_dims':   256,
    'color':          False,  # xyz のみ (C=3)
    'num_classes':    16,
}
# 投影ヘッド: PatchToTextProj(384 → 1280)
# CLIP: ViT-bigG-14 / laion2b_s39b_b160k  テキスト次元 = 1280
```

### forward_patches 出力仕様
```python
patch_emb, patch_centers, patch_idx = model.forward_patches(points)
# points       : (B, 3, N=2048)
# patch_emb    : (B, 384, G=128)
# patch_centers: (B, 3, G=128)
# patch_idx    : (B, G=128, M=32)
```

---

## コード構造 (Phase3_Training/)

```
Phase3_Training/
├── phase3_datasets/
│   └── dental_dataset.py    # DentalPatchDataset
├── phase3_models/
│   ├── stage3_model.py      # build_stage3_models(), PatchToTextProj
│   ├── text_cache.py        # build_clip_and_cache()
│   └── ewc.py               # EWC, compute_fisher_from_loader()
├── phase3_losses/
│   ├── infonce.py           # Stage 3a 損失
│   ├── bce_multilabel.py    # Stage 3b 損失
│   └── distillation.py      # KD 損失
├── train_stage3a.py
├── train_stage3b.py
├── configs/
│   ├── stage3a.yaml         # ローカル実行用 (相対パス)
│   └── stage3b.yaml
├── phase3_training.ipynb    # Colab 学習 notebook
├── phase3_inference.ipynb   # Colab 推論・可視化 notebook
└── outputs/
    ├── stage3a/
    └── stage3b/
```

---

## Colab セットアップ手順 (次回)

### 1. パッケージインストール
```bash
pip install open_clip_torch>=2.24.0 timm easydict termcolor ninja open3d plyfile pyyaml imageio -q
```

### 2. pointnet2_ops ビルド (毎回必要)
```bash
git clone --quiet --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/Pointnet2_PyTorch
find /tmp/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/src -type f \
    -exec sed -i 's/AT_CHECK/TORCH_CHECK/g' {} +
find /tmp/Pointnet2_PyTorch/pointnet2_ops_lib/pointnet2_ops/_ext-src/src -type f \
    -exec sed -i '/#include <THC\/THC.h>/d' {} +
# GPU アーキテクチャをセット (A100 = 8.0, T4 = 7.5)
major, minor = torch.cuda.get_device_capability()
# setup.py の TORCH_CUDA_ARCH_LIST を書き換えてから:
cd /tmp/Pointnet2_PyTorch/pointnet2_ops_lib && pip -v install .
```

### 3. sys.path & patchalign3d 仮想パッケージ登録
```python
import sys, types

PA3D_SRC  = '/content/drive/MyDrive/2026研究/ULIP_PointLLM/_reference_repos/PatchAlign3D/src'
PHASE3_DIR = '/content/drive/MyDrive/2026研究/ULIP_PointLLM/DentalPatchAligned3D/Phase3_Training'

for p in [PA3D_SRC, PHASE3_DIR]:
    if p not in sys.path:
        sys.path.insert(0, p)

if 'patchalign3d' not in sys.modules:
    mod = types.ModuleType('patchalign3d')
    mod.__path__ = [PA3D_SRC]
    sys.modules['patchalign3d'] = mod
```

---

## トラブルシューティング (今回発生した問題と解決策)

| エラー | 原因 | 解決策 |
|--------|------|--------|
| `AttributeError: module 'plyfile' has no __version__` | plyfile は `__version__` を持たない | `getattr(plyfile, '__version__', 'installed')` |
| `ModuleNotFoundError: No module named 'pointnet2_ops'` | `TORCH_CUDA_ARCH_LIST` 未設定で nvcc がクラッシュ | `setup.py` の arch を現在の GPU に書き換え後 `pip install .` |
| `ModuleNotFoundError: No module named 'datasets.dental_dataset'` | HuggingFace `datasets` パッケージと名前衝突 | `datasets/` → `phase3_datasets/` にリネーム |
| `ModuleNotFoundError: No module named 'models.stage3_model'` | `models` パッケージ名衝突 | `models/` → `phase3_models/` にリネーム |
| `ModuleNotFoundError: No module named 'patchalign3d'` | `patchalign3d/` ディレクトリが存在しない (仮想パッケージ) | `sys.modules` に手動登録 |
| `ModuleNotFoundError: No module named 'DentalPatchAligned3D'` | ewc.py / bce_multilabel.py にハードコードされた絶対パス | `from phase3_losses.xxx import` に修正 |
| `UnpicklingError: Weights only load failed` | PyTorch 2.6 で `torch.load` デフォルトが `weights_only=True` に変更 | `torch.load(..., weights_only=False)` |
| `sys.path.insert(0, _HERE.parent.parent)` が効かない | Phase3_Training の親ではなく ULIP_PointLLM が追加されていた | `sys.path.insert(0, str(_HERE))` (Phase3_Training 自身) |

---

## 設計メモ

### Stage 3a vs Stage 3b
- **Stage 3a**: 9 octant (LPS 方位) の粗い領域 × InfoNCE。candidates.json のテキスト**不使用**。
- **Stage 3b**: 32 パッチ × 5 テキスト (candidates.json) × BCE。本家 PatchAlign3D Stage 2 と同一手法。

### BCE はクラス分類ではない
```
Point Encoder → proj(384→1280) ─┐
                                  ├─ cosine similarity → logits → BCE(soft target)
CLIP Text Encoder(1280) ─────────┘
```
テキストは CLIP で動的にエンコードされる意味的参照ベクトル。固定クラス重みは**なし**。
BCE のターゲット `Y_patch` は「パッチ内に各ラベルの点が何割含まれるか」のソフト分布。

### カラーマップ
- 類似度ヒートマップは `coolwarm` を使用: **赤=高類似度、青=低類似度**
- `'hot'` は赤=高・黒=低 (直感に合わない場合がある)
