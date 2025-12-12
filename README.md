# AgentX42
# Identity-Employees-in-Surveillance-CCTV  
**Kaggle Competition Solution – EfficientNet-B0 + ArcFace**

A complete, **reproducible pipeline** for detecting and identifying authorized employees in CCTV frames.  
The notebook trains an **EfficientNet-B0 backbone with an ArcFace head**, augments the provided reference photos with video-frame extractions, filters by face quality, grid-searches optimal thresholds, and finally produces a `submission.csv`.

---

## Table of Contents
1. [Project Overview](#project-overview)  
2. [Directory Layout](#directory-layout)  
3. [Quick Start](#quick-start)  
4. [Methodology](#methodology)  
5. [Training & Validation](#training--validation)  
6. [Inference & Submission](#inference--submission)  
7. [Pre-trained Weights](#pre-trained-weights)  `
8. [External Assets & Licences](#external-assets--licences)  
9. [Reproducibility Checklist](#reproducibility-checklist)  
10. [Acknowledgements](#acknowledgements)  

---

## Project Overview
The goal is to **match every face in CCTV frames to a known employee ID** or label it as `"unknown"`.  
Key steps:

| Stage | What happens | Cells |
|-------|--------------|-------|
| **Data sanity & cleanup** | Drop broken rows; verify all image paths. | 1 |
| **Frame extraction** | Sample 500 evenly spaced frames per employee from reference videos. | A |
| **Dataset assembly** | Merge CCTV images, static photos & extracted frames; stratified split. | 2 |
| **Model training** | EfficientNet-B0 backbone → 512-d projection → ArcFace head (70 epochs). | 3 |
| **Gallery building** | Embed reference gallery, run FIQA quality filter, cosine threshold search. | 4 |
| **Threshold grid-search** | Jointly tune soft-max confidence `P_TH` and cosine `τ` for macro-accuracy. | 4B |
| **Hybrid inference** | Combine classifier & gallery for final predictions; save `submission.csv`. | 5 |

---

## Directory Layout
```
.
├── notebook.ipynb # this file – end-to-end pipeline
├── README.md # ← you are here
├── /kaggle/working
│   ├── effb0_arcface.pt # saved checkpoint
│   ├── gallery.pt # filtered embeddings + τ*
│   └── submission.csv # final output
└── /kaggle/input/identity-employees-in-surveillance-cctv
    └── dataset/… # competition data (train, reference_faces, unseen_test)
```

---

## Quick Start
```bash
# 1. Clone & open in Jupyter or Kaggle
git clone https://github.com/your-org/employee-cctv.git
cd employee-cctv

# 2. (Local) create environment
conda env create -f env.yml      # or pip install -r requirements.txt
conda activate cctv

# 3. Download competition data into /kaggle/input
kaggle competitions download -c identity-employees-in-surveillance-cctv
unzip identity-employees-in-surveillance-cctv.zip -d kaggle/input

# 4. Launch notebook and Run-All
jupyter lab
```

Tip: On Kaggle, simply “Copy & Edit” and hit Save & Run All – GPU T4 runtime is automatically provisioned.

## Methodology
Aggressive augmentation on CCTV images (crop, JPEG artefacts, blur, perspective, erasing) to mimic surveillance variability.

ArcFace head provides angular margin for better class separation; cosine similarities are later reused for gallery matching.

Hybrid decision rule

If soft-max probability ≥ P_TH → trust classifier.

Else, fall back to gallery cosine ≥ τ.

Otherwise label "unknown".

FIQA filtering removes the noisiest 20 % of gallery faces, boosting similarity reliability.

Grid search over (P_TH, τ) on the held-out CCTV split maximises macro-accuracy, the competition metric.

## Training & Validation

```bash
# default: 70 epochs, batch=32, lr=3e-4 (AdamW), cosine schedule
python train.py --epochs 70 --batch 32 --lr 3e-4
Best held-out macro-accuracy: 0.81 (will vary ±0.01 due to randomness).

To reproduce exactly:

```bash
python train.py --seed 42 --epochs 70 --checkpoint effb0_arcface.pt
```

## Inference & Submission

```bash
python infer.py \
  --ckpt effb0_arcface.pt \
  --gallery gallery.pt \
  --p_th 0.58 \
  --tau 0.46 \
  --out submission.csv
```
Outputs a submission.csv ready for Kaggle upload:

```csv
image_name,employee_id
frame_000001.jpg,emp017
frame_000002.jpg,unknown
...
```
## Pre-trained Weights

| File | Size | Notes |
|------|------|-------|
| effb0_arcface.pt | 14 MB | Backbone + projection + ArcFace (s≈30, m≈0.50) |
| gallery.pt | 24 MB | Normalised 512-d embeddings + label list + best τ |

Download from the releases page or regenerate with the notebook.

## External Assets & Licences

| Asset | Purpose | Licence |
|-------|---------|---------|
| EfficientNet-B0 weights (timm) | Image backbone | Apache 2.0 |
| InsightFace FIQA (buffalo_s) | Face quality assessment | MIT |
| Competition dataset | CCTV frames & reference faces | © Abu Dhabi 42 (competition rules) |

Project code is released under the MIT License (OSI-approved).
See LICENSE for details.

## Reproducibility Checklist

- [x] All random seeds fixed (numpy, torch, random)
- [x] Exact package versions pinned in env.yml
- [x] Training + inference scripts runnable end-to-end
- [x] Pre-trained checkpoint and gallery embeddings provided
- [x] No internet access required after data download

## Acknowledgements

- Kaggle & 42 Abu Dhabi for hosting the dataset

# data-mind-cv
# data-mind-cv
