# Google Colab 実行ガイド — Phase 0 CLIP モード

## MPS 対応状況

| 環境 | 状況 | 備考 |
|------|------|------|
| CUDA (Colab T4/L4/A100) | **推奨** | 安定・高速 |
| MPS (Apple M1/M2/M3) | 動作するが非推奨 | ViT-bigG-14 (~1.9B params) は VRAM 不足・精度低下リスクあり |
| CPU | 動作するが低速 | 1症例あたり数時間かかる |

open_clip は標準 PyTorch 演算のみ使用するため MPS バックエンドで動作する。
ただし PyTorch MPS では一部 reduce 演算が float32 にアップキャストされ、
余剰メモリを消費する場合がある。Step 1 は Colab で実行することを推奨する。


## ワークフロー

```
[Colab]  run_phase0_clip_colab.ipynb  Step 1 セル   → candidates_clip.json + renders/clip/
                                                          ↓ (Google Drive に自動保存)
[ローカル]  python run_phase0.py --mode clip --step 2  → verified_labels_clip.json
                                                          ↓ (Google Drive に保存後)
[Colab または ローカル]  Step 3 セル / run_phase0.py --mode clip --step 3
                                                          → mask2points.pt / mask_labels.txt
```


## Colab 実行手順

### 事前準備

1. `ランタイム > ランタイムのタイプを変更` → **GPU** (T4 以上を推奨)
2. Google Drive に PLY ファイルが配置されていることを確認
   - パス: `マイドライブ/2026研究/Dataset/6167726/STLs/PLYs/*.ply`

### セル実行順序

| # | 内容 | 所要時間目安 |
|---|------|-------------|
| 1 | Drive マウント | 数秒 |
| 2 | パッケージインストール | 1〜2 分 |
| 3 | sys.path 設定 | 即時 |
| 4 | デバイス確認 | 即時 |
| 5 | パラメータ確認 | 即時 |
| 6 | **Step 1 実行** (CLIP ランキング) | T4 で 1 症例あたり 5〜15 分 |
| 7 | Step 3 事前チェック | 即時 |
| 8-9 | Step 3 実行 (Step 2 完了後) | 数分 |

### パラメータ変更

ノートブック Step 1 パラメータセル (`N_PATCHES` 等) を直接編集する。


## ローカル実行手順 (Step 2)

Colab で Step 1 完了後、Google Drive に `candidates_clip.json` と
`renders/clip/` が生成される。ローカルで以下を実行する:

```bash
cd /path/to/ULIP_PointLLM/DentalPatchAligned3D/Phase0_Data

python run_phase0.py --mode clip --step 2
# オプション:
#   --no-open3d   Open3D ウィンドウをスキップ
#   --no-images   画像自動表示をスキップ
```

### 確認 CLI の操作

```
[1〜N]  その番号のテキストを採用 (複数可: "135" で 1,3,5 番を採用)
[a]     全候補を採用
[eN]    N 番のテキストを編集 (例: e2)
[r]     再生成メモを入力してスキップ
[x]     このパッチを除外
[AA]    残り全パッチを全採用してスキップ (確認プロンプトあり)
```

CLIP モードでは `texts` フィールドに cosine 類似度上位 N 件の短いラベルが並ぶ。
`clip_scores` フィールドのスコアは表示されないが `candidates_clip.json` で確認できる。


## ファイル出力先 (Google Drive)

```
2026研究/ULIP_PointLLM/data/DentalPatchData/
├── dental_vocab/
│   ├── candidates_clip.json        ← Step 1 出力
│   ├── verified_labels_clip.json   ← Step 2 出力 (ローカル)
│   └── anatomy_textbank.pt         ← Step 3 出力
├── renders/clip/
│   └── <case_id>/patch_NNN/
│       ├── az000.png  (患者左側)
│       ├── az045.png  (左後斜め)
│       ├── az090.png  (後方)
│       ├── az135.png  (右後斜め)
│       ├── az180.png  (患者右側)
│       ├── az225.png  (右前斜め)
│       ├── az270.png  (前方)
│       └── az315.png  (左前斜め)
└── dental_dataset/labeled/         ← Step 3 出力
    ├── points/
    ├── rendered/
    └── split/
```


## トラブルシューティング

| 症状 | 対処 |
|------|------|
| `CUDA out of memory` | `N_PATCHES` を減らす / ランタイムを再起動してバッチを小さくする |
| `open_clip not found` | Cell 2 (pip install) を再実行 |
| `PLY が見つかりません` | `cfg.PLY_DIR` のパスを確認 (Drive マウント確認) |
| Step 1 途中で切断 | `candidates_clip.json` に途中結果が保存されているため再実行すると再開する |
| MPS で `MPS backend out of memory` | `--device cpu` に変更するか Colab を使用 |
