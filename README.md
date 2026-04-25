# patch-align-mandible

**Text-guided 3D patch alignment for mandibular anatomy localization**

A two-stage contrastive learning framework that aligns local surface patches of mandible point clouds with anatomical text descriptions. Given a query such as *"left mandibular condyle"*, the model highlights the corresponding region on a 3D scan.

---

## Demo

| Cosine Similarity Map | Anatomy Search |
|:---------------------:|:--------------:|
| ![CosSim](assets/CosSim.png) | ![Demo](assets/example.gif) |

### Text label example — `Pat 1a_norm / patch 06` (left mandibular condyle)

Each surface patch is described by five LLM-generated anatomy sentences used as contrastive text targets during training.

<p align="center">
  <img src="assets/LeftCoronoidProcessPatch.png" width="340" alt="Left mandibular condyle patch (Pat 1a_norm patch06)"/>
</p>

| # | Generated text |
|:-:|----------------|
| 1 | Left mandibular condylar process, rounded articular head with smooth convex surface. |
| 2 | Left condyle of mandible, the superior convex process articulating at the TMJ. |
| 3 | Left mandibular head, rounded and expanded articular surface at the ramus apex. |
| 4 | Left condylar process, oval superior projection forming the mandibular articular surface. |
| 5 | Left mandibular condyle, a broad convex process on the posterior ramus end. |

---

## Repository Structure

```
patch-align-mandible/
├── data_pipeline/          # CT → point cloud preprocessing & LLM-assisted text labelling
├── training/               # Stage 3a + 3b training  (ViT-bigG-14, base)
├── training_biomedclip/    # Stage 3b training        (BiomedCLIP backbone)
├── training_text_tune/     # Stage 3a + 3b training   (text encoder fine-tuned)
├── apple_deploy/           # MLX / CoreML / Swift deployment for Apple Silicon
├── assets/                 # Images and animations used in this README
├── requirements.txt
└── README.md
```

---

## Data Pipeline (`data_pipeline/`)

Converts mandible CT meshes into labelled point cloud datasets ready for contrastive training.

```
data_pipeline/
├── config.py                                    # Global path configuration
├── run_phase0.py                                # End-to-end pipeline entry point
├── run_phase0_clip_colab.ipynb                  # Colab notebook version
├── tools/
│   ├── mesh_to_pointcloud.py                    # STL/OBJ → PLY (normalised, 2048 pts)
│   ├── build_anatomy_textbank.py                # Builds anatomy term vocabulary
│   ├── generate_text_candidates.example.py      # Template — copy before use (see below)
│   ├── clip_text_ranker.py                      # CLIP-based text candidate ranking
│   ├── build_patchalign_dataset.py              # Assembles final dataset JSON
│   └── verify_text_labels.py                   # Sanity-check label coverage
└── prompts/
    ├── system_prompt.txt                        # LLM system prompt for anatomy description
    └── user_prompt_template.json                # Per-patch query template
```

### Setting up the LLM text generator

`generate_text_candidates.py` calls an external LLM API and is excluded from the repository.
Copy the example template and register your API key before running:

```bash
cp data_pipeline/tools/generate_text_candidates.example.py \
   data_pipeline/tools/generate_text_candidates.py

# Open the file and set your API key:
#   ANTHROPIC_API_KEY = "sk-ant-..."   # Anthropic Claude
#   or
#   OPENAI_API_KEY = "sk-..."          # OpenAI
```

---

## Contrastive Training (`training*/`)

### Model Architecture

![Model Overview](assets/model_overview.png)

The encoder is built on **PatchAlign3D** (Point Transformer with FPS + KNN grouping).
Training proceeds in two stages:

**Stage 3a — Domain-level InfoNCE**
The full point cloud is divided into local patches. A patch feature vector is pulled toward its matched anatomy text embedding (CLIP-encoded) and pushed away from unmatched ones via InfoNCE contrastive loss. This stage establishes coarse anatomical grounding.

**Stage 3b — Local patch BCE + Knowledge Distillation + EWC**
Each patch is classified against a fixed set of 16 anatomy labels with binary cross-entropy. Knowledge Distillation (KD) preserves the CLIP teacher's soft targets, and Elastic Weight Consolidation (EWC) prevents catastrophic forgetting of Stage 3a representations. Text augmentation selects one of five paraphrases per label per epoch.

### Variant Comparison

| Folder | CLIP Backbone | Text Dim | Stages | Key Difference |
|--------|--------------|:--------:|:------:|----------------|
| `training/` | ViT-bigG-14 (LAION-2B) | 1280 | 3a → 3b | Base pipeline |
| `training_biomedclip/` | BiomedCLIP (MS PubMedBERT + ViT-B/16) | 512 | 3b only | Skips Stage 3a; initialises from a Stage 2 checkpoint; biomedical language model |
| `training_text_tune/` | ViT-bigG-14 / BiomedCLIP | 1280 / 512 | 3a → 3b | Text encoder fine-tuned jointly with the point encoder |

### Running Training (Colab)

Each training folder contains a ready-to-run Colab notebook:

```
training/phase3_training.ipynb
training_biomedclip/phase3_training.ipynb
training_text_tune/phase3_training_bigG14.ipynb
```

Or run from the command line:

```bash
cd training/
python train_stage3a.py --config configs/stage3a.yaml
python train_stage3b.py --config configs/stage3b.yaml \
    --stage3a_ckpt outputs/stage3a/stage3a_last.pt
```

---

## Apple Silicon Deployment (`apple_deploy/`)

Trained weights are converted to run natively on Apple Silicon via **MLX** (Mac demo) and
**CoreML + Swift** (iOS / visionOS). A visionOS immersive-space app (`PatchSimlilarySpace`)
visualises anatomy search results on a 3D mandible model in real time.

| Component | Description |
|-----------|-------------|
| Python + MLX | Mac inference demo — anatomy search from a PLY file |
| `DentalInferenceKit` | Swift Package wrapping the CoreML point encoder and CLIP text encoder |
| `PatchSimlilarySpace` | visionOS app — immersive 3D anatomy viewer for Apple Vision Pro |

[![visionOS App Demo](https://img.youtube.com/vi/dWb9BIXHC7Q/maxresdefault.jpg)](https://youtu.be/dWb9BIXHC7Q)

For the full weight conversion pipeline, file placement guide, and Xcode setup instructions,
see **[`apple_deploy/README.md`](apple_deploy/README.md)**.

---

## Getting Started

### Prerequisites

- Python 3.10 or 3.11
- CUDA 11.8+ GPU (training) — Google Colab A100 recommended
- Apple Silicon Mac (deployment, optional)

### Install

```bash
git clone https://github.com/<your-username>/patch-align-mandible.git
cd patch-align-mandible

pip install -r requirements.txt

# Additional manual installs:
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"
pip install https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Dataset

Download the Mandibular CT Dataset from Figshare and extract STL files to
`Dataset/6167726/STLs/STLs/` relative to the project root:

```
https://doi.org/10.6084/m9.figshare.6167726.v6
```

Then run the data pipeline:

```bash
python data_pipeline/run_phase0.py
```

---

## References

**Backbone**

```bibtex
@misc{hadgi2026patchalign3dlocalfeaturealignment,
  title         = {PatchAlign3D: Local Feature Alignment for Dense 3D Shape understanding},
  author        = {Souhail Hadgi and Bingchen Gong and Ramana Sundararaman and Emery Pierson
                   and Lei Li and Peter Wonka and Maks Ovsjanikov},
  year          = {2026},
  eprint        = {2601.02457},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2601.02457},
}
```

**CLIP backbone (ViT-bigG-14) pre-training data**

```bibtex
@inproceedings{schuhmann2022laionb,
  title     = {{LAION}-5B: An open large-scale dataset for training next generation image-text models},
  author    = {Christoph Schuhmann and Romain Beaumont and Richard Vencu and Cade W Gordon
               and Ross Wightman and Mehdi Cherti and Theo Coombes and Aarush Katta
               and Clayton Mullis and Mitchell Wortsman and Patrick Schramowski
               and Srivatsa R Kundurthy and Katherine Crowson and Ludwig Schmidt
               and Robert Kaczmarczyk and Jenia Jitsev},
  booktitle = {Thirty-sixth Conference on Neural Information Processing Systems Datasets and Benchmarks Track},
  year      = {2022},
  url       = {https://openreview.net/forum?id=M3Y74vmsMcY},
}
```

**OpenCLIP**

```bibtex
@software{ilharco_gabriel_2021_5143773,
  author    = {Ilharco, Gabriel and Wortsman, Mitchell and Wightman, Ross and Gordon, Cade
               and Carlini, Nicholas and Taori, Rohan and Dave, Achal and Shankar, Vaishaal
               and Namkoong, Hongseok and Miller, John and Hajishirzi, Hannaneh
               and Farhadi, Ali and Schmidt, Ludwig},
  title     = {OpenCLIP},
  month     = jul,
  year      = 2021,
  publisher = {Zenodo},
  version   = {0.1},
  doi       = {10.5281/zenodo.5143773},
  url       = {https://doi.org/10.5281/zenodo.5143773},
}
```

**BiomedCLIP (alternative text encoder)**

```bibtex
@misc{zhang2023biomedclip,
  title         = {BiomedCLIP: a multimodal biomedical foundation model pretrained from
                   fifteen million scientific image-text pairs},
  author        = {Sheng Zhang and Yanbo Xu and Naoto Usuyama and Jaspreet Bagga
                   and Robert Tinn and Sam Preston and Rajesh Rao and Mu Wei
                   and Naveen Vajjala and Subhashini Venugopalan and Chitta Baral
                   and Xin Liu and Matthew P. Lungren and Tristan Naumann
                   and Chunyuan Li and Hoifung Poon},
  year          = {2023},
  eprint        = {2303.00915},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CV},
  url           = {https://arxiv.org/abs/2303.00915},
}
```

**ULIP (3D–language pre-training)**

```bibtex
@inproceedings{xue2023ulip,
  title     = {{ULIP}: Learning a Unified Representation of Language, Images, and Point Clouds
               for 3D Understanding},
  author    = {Le Xue and Mingfei Gao and Chen Xing and Roberto Martín-Martín and Jiajun Wu
               and Caiming Xiong and Ran Xu and Juan Carlos Niebles and Silvio Savarese},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages     = {1179--1189},
  year      = {2023},
}
```

**PointLLM (3D–language pre-training)**

```bibtex
@article{xu2023pointllm,
  title   = {PointLLM: Empowering Large Language Models to Understand Point Clouds},
  author  = {Runsen Xu and Xingrui Wang and Tai Wang and Yilun Chen
             and Jiangmiao Pang and Dahua Lin},
  journal = {arXiv preprint arXiv:2308.16911},
  year    = {2023},
  url     = {https://arxiv.org/abs/2308.16911},
}
```

**Dataset**

```bibtex
@data{wallner2018mandible,
  title     = {Mandibular CT Dataset Collection},
  author    = {Wallner, Jürgen and Egger, Jan},
  publisher = {figshare},
  year      = {2018},
  doi       = {10.6084/m9.figshare.6167726.v6},
  url       = {https://doi.org/10.6084/m9.figshare.6167726.v6},
}
```
