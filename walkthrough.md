# Walkthrough: Autoencoders for Open-Set PAD

## What Was Built

A complete end-to-end pipeline for **Autoencoders for Open-Set Presentation Attack Detection** using the **RECOD-MPAD** dataset (26 GB), running on a local machine with RTX 4050 (6 GB VRAM).

## Project Structure

```
d:\DAU\Sem_2\Biometric Security\Project\
├── src/                          ← Core Python modules
│   ├── config.py                 ← All hyperparameters & paths
│   ├── dataset.py                ← Manifest builder + PyTorch Dataset  
│   ├── dwt.py                    ← 2D Discrete Wavelet Transform
│   ├── model.py                  ← Encoder + Denoiser + SpectralDecoder
│   ├── losses.py                 ← Combined MSE + L1 loss
│   ├── train.py                  ← Training loop (FP16, early stopping)
│   ├── evaluate.py               ← Open-set evaluation & metrics
│   └── utils.py                  ← Visualization & utilities
├── notebooks/                    ← Jupyter notebooks (run in order)
│   ├── 01_data_exploration.ipynb ← Dataset stats & DWT visualization
│   ├── 02_preprocessing.ipynb    ← Build manifest, verify pipeline
│   ├── 03_model_training.ipynb   ← Train the autoencoder
│   ├── 04_evaluation.ipynb       ← Open-set evaluation & metrics
│   └── 05_results_analysis.ipynb ← t-SNE, ablation, confusion matrix
└── requirements.txt              ← Python dependencies
```

## Architecture (Matching Workflow Diagram)

The model follows exactly the workflow provided:

1. **2D-DWT** decomposes face images into LL, LH, HL, HH sub-bands (Haar wavelet)
2. **Lightweight CNN Encoder** (2.4M params) maps concatenated bands → latent z₀ (512-d)
3. **Forward Diffusion** adds controlled Gaussian noise: z₀ → z_T*
4. **Denoiser MLP** (2.1M params) with residual connections reverses: z_T* → ẑ₀
5. **Spectral Decoder** (1.8M params) reconstructs HH band from ẑ₀
6. **Dual Scoring**: S_latent (MSE) + S_spectral (L1 on HH)
7. **Weighted Fusion**: S_final = λ·S_latent + (1-λ)·S_spectral
8. **Decision**: S_final > τ → Attack / Bona Fide

**Total: ~6.3M parameters (~25 MB)** — uses only ~1.1 GB of 6 GB VRAM.

## Open-Set Protocol

- **Train**: Bona fide only (users 1-30) — autoencoder learns "normal" face distribution
- **Validation**: Bona fide + Print Indoor attacks (users 31-37) — for threshold τ tuning
- **Test**: Bona fide + ALL 4 attack types (users 38-45) — including 3 UNSEEN attack types
- User-disjoint splits ensure no identity leakage

## How to Run

### Step 1: Install dependencies
```bash
cd "d:\DAU\Sem_2\Biometric Security\Project"
pip install -r requirements.txt
```

### Step 2: Run notebooks in order (1 → 5) in Jupyter
```bash
jupyter notebook notebooks/
```

Each notebook imports from `src/` and is self-contained with inline visualizations.

## Key Design Decisions

| Decision | Rationale |
|---|---|
| Train on bona fide only | Ensures open-set generalization — no attack samples needed |
| 128×128 input resolution | Sweet spot for quality vs. VRAM on 6 GB GPU |
| Every 5th frame sampling | Reduces data 5× while keeping diversity |
| FP16 mixed precision | Cuts VRAM usage ~50%, 2× speedup |
| Haar wavelet | Simplest DWT, effective for artifact detection |
| Dual scoring (MSE + L1) | Latent captures global anomalies; HH captures frequency artifacts |
