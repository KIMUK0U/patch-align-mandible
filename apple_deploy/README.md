# Apple Silicon Deployment

Covers three layers of deployment built on top of the trained patch encoder:

| Layer | Tool | Target |
|-------|------|--------|
| Mac inference / demo | Python + MLX | Apple Silicon Mac |
| Inference library | CoreML + Swift Package (`DentalInferenceKit`) | iOS 17+ / visionOS 1+ / macOS 14+ |
| visionOS app | Xcode (`PatchSimlilarySpace`) | Apple Vision Pro |

---

## Directory Layout

```
apple_deploy/
├── convert_weights.py          # Step 1 — PyTorch .pt → MLX .npz
├── precompute_text_emb.py      # Step 2 — pre-compute text embeddings (.npz)
├── inference_mac.py            # Mac demo — anatomy search from a PLY file
├── export_coreml.py            # Step 3 — point encoder → dental_patch_encoder.mlpackage
├── save_text_emb_bin.py        # Step 4 — .npz embeddings → .bin for Swift
├── model/
│   ├── mlx_point_transformer.py   # PatchAlign3D encoder in MLX
│   └── point_ops.py               # FPS + KNN (NumPy, no CUDA needed)
├── clip_text/
│   ├── extract_clip_weights.py
│   ├── export_clip_coreml.py      # Step 5 — CLIP text encoder → dental_clip_text.mlpackage
│   ├── mlx_clip_text.py
│   ├── save_clip_tokenizer.py     # Step 6 — tokenizer vocab/merges for Swift
│   └── search_mac.py              # Interactive anatomy search (Mac)
├── DentalInferenceKit/            # Swift Package — add as local dep to Xcode project
│   └── Sources/DentalInferenceKit/
│       ├── CLIPTokenizer.swift
│       ├── CLIPTextEncoder.swift
│       ├── DentalInference.swift
│       ├── PatchSearchEngine.swift
│       ├── PointOps.swift
│       └── Resources/
│           ├── clip_vocab.json                    ← tracked in git
│           ├── clip_merges.txt                    ← tracked in git
│           ├── dental_clip_text.mlpackage/        ← place here (Step 5, git-ignored)
│           ├── dental_patch_encoder.mlpackage/    ← place here (Step 3, git-ignored)
│           └── text_embeddings.bin                ← place here (Step 4, git-ignored)
└── PatchSimlilarySpace/           # visionOS Xcode app
    └── PatchSimlilarySpace/
        ├── *.swift
        ├── Mandible.usdz          ← place here (git-ignored, provide separately)
        └── ...
```

> **Note on git-ignored files**  
> Weight files (`.npz`, `.pt`, `.bin`), CoreML models (`.mlpackage/`), and 3D assets (`.usdz`)
> are all excluded from the repository. Follow the steps below to regenerate them.

---

## Step-by-Step: Building Weights for Deployment

### Prerequisites

```bash
pip install mlx open_clip_torch coremltools transformers
```

---

### Step 1 — Convert PyTorch checkpoint to MLX

Run after completing Stage 3b training in `training/`.

```bash
python apple_deploy/convert_weights.py \
    --ckpt training/outputs/stage3b/stage3b_best.pt
# → apple_deploy/outputs/npz/phase3_weights.npz  (directory auto-created)
```

---

### Step 2 — Pre-compute text embeddings (Mac / MLX)

```bash
python apple_deploy/precompute_text_emb.py
# → apple_deploy/outputs/npz/text_embeddings.npz  (directory auto-created)
```

Optional — verify on a real scan with the Mac demo:

```bash
python apple_deploy/inference_mac.py \
    --weights   apple_deploy/outputs/npz/phase3_weights.npz \
    --ply       path/to/scan.ply \
    --text_emb  apple_deploy/outputs/npz/text_embeddings.npz
```

---

### Step 3 — Export point encoder to CoreML

```bash
python apple_deploy/export_coreml.py \
    --ckpt training/outputs/stage3b/stage3b_best.pt \
    --out  apple_deploy/dental_patch_encoder.mlpackage

cp -r apple_deploy/dental_patch_encoder.mlpackage \
      apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/
```

---

### Step 4 — Convert text embeddings to Swift binary

```bash
python apple_deploy/save_text_emb_bin.py \
    --out apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/text_embeddings.bin
# --npz defaults to outputs/npz/text_embeddings.npz
```

---

### Step 5 — Export CLIP text encoder to CoreML

```bash
python apple_deploy/clip_text/extract_clip_weights.py
python apple_deploy/clip_text/export_clip_coreml.py \
    --out apple_deploy/dental_clip_text.mlpackage

cp -r apple_deploy/dental_clip_text.mlpackage \
      apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/
```

---

### Step 6 — Save tokenizer (one-time setup)

`clip_vocab.json` and `clip_merges.txt` are already committed to the repository.
Re-run only if you switch the CLIP backbone:

```bash
python apple_deploy/clip_text/save_clip_tokenizer.py \
    --out_dir apple_deploy/DentalInferenceKit/Sources/DentalInferenceKit/Resources/
```

---

## Resources Checklist

Verify all files are in place before building the Xcode project:

```
DentalInferenceKit/Sources/DentalInferenceKit/Resources/
├── clip_vocab.json                    ✅ in git
├── clip_merges.txt                    ✅ in git
├── dental_clip_text.mlpackage/        ← Step 5
├── dental_patch_encoder.mlpackage/    ← Step 3
└── text_embeddings.bin                ← Step 4

PatchSimlilarySpace/PatchSimlilarySpace/
└── Mandible.usdz                      ← provide your own (convert STL via Reality Converter)
```

---

## visionOS App (`PatchSimlilarySpace/`)

Open `PatchSimlilarySpace/PatchSimlilarySpace.xcodeproj` in Xcode 15+.

**1. Add `DentalInferenceKit` as a local Swift Package dependency:**

- File → Add Package Dependencies → Add Local…
- Select `apple_deploy/DentalInferenceKit/`
- Add `DentalInferenceKit` to the `PatchSimlilarySpace` target

**2. Place the 3D model:**

Place `Mandible.usdz` inside `PatchSimlilarySpace/PatchSimlilarySpace/`.  
The model is referenced in `ImmersiveView.swift` and rendered in the visionOS immersive space.

**3. Build and run** on Apple Vision Pro or the visionOS Simulator (Xcode 15+, visionOS SDK required).

---

## Swift API (DentalInferenceKit)

```swift
import DentalInferenceKit

// Text-guided anatomy search — returns per-patch similarity scores
let results = try await DentalInference.shared.search(
    xyz: flatXYZ,                     // [N × 3] Float32, row-major
    query: "left mandibular condyle"
)

// Full classification — returns top anatomy label per patch
let labels = try await DentalInference.shared.predict(xyz: flatXYZ)
```

### Design notes

- **FPS + KNN** is implemented in pure Swift using the Accelerate framework (`PointOps.swift`)
  because CoreML cannot handle dynamic graph structures at runtime.
- **Neural Engine** is used for the Encoder + Transformer + Projection layers via CoreML.
- No Objective-C or C++ bridging required.
