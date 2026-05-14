# %% [markdown]
# # Notebook 5: Results Analysis & Visualization
# **Autoencoders for Open-Set Presentation Attack Detection**
# 
# This notebook provides detailed analysis:
# - Latent space t-SNE visualization
# - HH band reconstruction comparison (real vs. attack)
# - Lambda (λ) ablation study for fusion weight
# - Confusion matrices
# - Final publication-quality figures

# %% [markdown]
# ## 1. Setup

# %%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, roc_curve, auc as sk_auc

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config as cfg
from src.dataset import load_manifest, create_dataloaders
from src.model import build_model
from src.evaluate import AOPADEvaluator
from src.utils import (set_seed, plot_latent_tsne, plot_hh_reconstruction)

set_seed()

# %% [markdown]
# ## 2. Load Model & Data

# %%
model = build_model()
checkpoint = torch.load(
    os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'),
    map_location=cfg.DEVICE,
)
model.load_state_dict(checkpoint['model_state_dict'])
model = model.to(cfg.DEVICE)
model.eval()
print(f"✓ Model loaded (epoch {checkpoint['epoch']+1})")

manifest = load_manifest()
train_loader, val_loader, test_loader = create_dataloaders(manifest)

# Load saved scores (produced by 04_evaluation.ipynb)
test_scores = dict(np.load(os.path.join(cfg.RESULTS_DIR, 'test_scores.npz')))

# %% [markdown]
# ## 3. Latent Space t-SNE Visualization
# Shows how real and attack faces cluster in the learned latent space.

# %%
# Collect latent vectors from test set
all_latents = []
all_labels = []

with torch.no_grad():
    for dwt_concat, hh_band, labels, metadata in tqdm(test_loader, desc="Extracting latents"):
        dwt_concat = dwt_concat.to(cfg.DEVICE)
        with autocast(enabled=cfg.USE_AMP):
            z0 = model.encoder(dwt_concat)
        all_latents.append(z0.cpu().numpy())
        all_labels.append(labels.numpy())

latent_vectors = np.concatenate(all_latents)
labels_array = np.concatenate(all_labels)

print(f"Latent vectors shape: {latent_vectors.shape}")
print(f"Labels distribution: {dict(zip(*np.unique(labels_array, return_counts=True)))}")

# %%
# Subsample for speed if needed (t-SNE is O(n²))
max_tsne_samples = 3000
if len(latent_vectors) > max_tsne_samples:
    idx = np.random.choice(len(latent_vectors), max_tsne_samples, replace=False)
    latent_sub = latent_vectors[idx]
    labels_sub = labels_array[idx]
else:
    latent_sub = latent_vectors
    labels_sub = labels_array

fig = plot_latent_tsne(
    latent_sub, labels_sub,
    save_path=os.path.join(cfg.RESULTS_DIR, 'tsne_latent.png')
)

# Inline display
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=cfg.SEED, perplexity=30)
embedded = tsne.fit_transform(latent_sub)

plt.figure(figsize=(10, 8))
colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6', '#3498db']
for lbl in range(5):
    mask = (labels_sub == lbl)
    if mask.sum() == 0:
        continue
    name = cfg.ATTACK_NAMES[lbl]
    plt.scatter(embedded[mask, 0], embedded[mask, 1],
                c=colors[lbl], label=name, alpha=0.5, s=20)
plt.title('Latent Space t-SNE: Bona Fide vs. Attacks', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 4. HH Band Reconstruction Visualization
# Compare original vs. reconstructed HH bands for real and attack images.

# %%
# Collect HH reconstructions from test set
all_hh_orig = []
all_hh_recon = []
all_hh_labels = []

with torch.no_grad():
    for dwt_concat, hh_band, labels, metadata in tqdm(test_loader, desc="Reconstructing HH"):
        dwt_concat = dwt_concat.to(cfg.DEVICE)
        hh_band_gpu = hh_band.to(cfg.DEVICE)
        
        with autocast(enabled=cfg.USE_AMP):
            B = dwt_concat.size(0)
            noise_level = torch.full((B,), cfg.NOISE_LEVEL, device=cfg.DEVICE)
            output = model(dwt_concat, hh_band_gpu, noise_level)
        
        all_hh_orig.append(hh_band.numpy())
        all_hh_recon.append(output['hh_recon'].cpu().float().numpy())
        all_hh_labels.append(labels.numpy())
        
        if len(all_hh_labels) >= 5:  # Only need a few batches
            break

hh_orig = np.concatenate(all_hh_orig)
hh_recon = np.concatenate(all_hh_recon)
hh_labels = np.concatenate(all_hh_labels)

# %%
# Select samples: 4 real + 4 attack
real_idx = np.where(hh_labels == 0)[0][:4]
attack_idx = np.where(hh_labels > 0)[0][:4]
selected_idx = np.concatenate([real_idx, attack_idx])

fig = plot_hh_reconstruction(
    hh_orig[selected_idx], hh_recon[selected_idx], 
    hh_labels[selected_idx], n_samples=8,
    save_path=os.path.join(cfg.RESULTS_DIR, 'hh_reconstruction.png')
)

# Inline
fig, axes = plt.subplots(3, 8, figsize=(24, 9))
for i, idx in enumerate(selected_idx):
    # Original HH
    orig = hh_orig[idx].transpose(1, 2, 0)
    orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
    axes[0, i].imshow(orig)
    label_name = cfg.ATTACK_NAMES[hh_labels[idx]]
    axes[0, i].set_title(label_name, fontsize=8)
    axes[0, i].axis('off')
    
    # Reconstructed HH
    recon = hh_recon[idx].transpose(1, 2, 0)
    recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
    axes[1, i].imshow(recon)
    axes[1, i].axis('off')
    
    # Difference (error map)
    diff = np.abs(orig - recon)
    axes[2, i].imshow(diff, cmap='hot')
    axes[2, i].axis('off')

axes[0, 0].set_ylabel('Original HH', fontsize=10, fontweight='bold')
axes[1, 0].set_ylabel('Reconstructed HH', fontsize=10, fontweight='bold')
axes[2, 0].set_ylabel('Error Map', fontsize=10, fontweight='bold')
plt.suptitle('HH Band Reconstruction: Real (left) vs Attack (right)', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# %% [markdown]
# ## 5. K-Cluster & Contamination Configuration
# Display the AOPAD ensemble parameters chosen automatically via Silhouette Score.

# %%
import json

with open(os.path.join(cfg.RESULTS_DIR, 'evaluation_results.json'), 'r') as f:
    eval_results = json.load(f)

print("AOPAD Ensemble Configuration (auto-selected via Silhouette Score):")
print(f"  Optimal K (clusters)  = {eval_results['optimal_k']}")
print(f"  PCA components kept   = {eval_results['pca_components']}")
print(f"  Contamination/cluster = {eval_results['optimal_contamination']}")

# Plot contamination per cluster
conts = eval_results['optimal_contamination']
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(range(len(conts)), conts, color='steelblue', edgecolor='white')
ax.set_xlabel('Cluster Index', fontsize=12)
ax.set_ylabel('Contamination Parameter', fontsize=12)
ax.set_title('AOPAD: Auto-Selected Contamination per Cluster', fontweight='bold')
ax.set_xticks(range(len(conts)))
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'contamination_per_cluster.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Confusion Matrix

# %%
# Load binary predictions saved by 04_evaluation.ipynb (AOPAD ensemble output)
test_npz = dict(np.load(os.path.join(cfg.RESULTS_DIR, 'test_scores.npz')))

labels    = test_npz['labels']
s_final   = test_npz['s_final']

# Derive binary predictions from continuous score (median split for confusion matrix)
# Note: AOPAD uses IF ensemble for hard decisions; here we use median of s_final
# for a visualisation-only confusion matrix (the real metrics are in eval_results.json)
threshold_vis = np.median(s_final)
predictions   = (s_final > threshold_vis).astype(int)
labels_binary = (labels > 0).astype(int)

cm = confusion_matrix(labels_binary, predictions, labels=[0, 1])

plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Bona Fide', 'Attack'],
            yticklabels=['Bona Fide', 'Attack'],
            annot_kws={'size': 16})
plt.title('Confusion Matrix (Test Set — median score threshold)', fontsize=13, fontweight='bold')
plt.xlabel('Predicted', fontsize=12)
plt.ylabel('Actual', fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'confusion_matrix.png'), dpi=150)
plt.show()

print(f"True Negatives (Correct Bona Fide):  {cm[0,0]}")
print(f"False Positives (Bona Fide -> Attack): {cm[0,1]}")
print(f"False Negatives (Attack -> Bona Fide): {cm[1,0]}")
print(f"True Positives (Correct Attack):      {cm[1,1]}")

# %% [markdown]
# ## 7. Final Summary

# %%
print("\n" + "="*70)
print("   FINAL RESULTS: Autoencoders for Open-Set PAD")
print("="*70)

overall = eval_results['test_metrics']['overall']
print(f"\n  Model:    DWT Denoising Autoencoder + AOPAD Cluster-Aware Ensemble")
print(f"  Dataset:  RECOD-MPAD (2020)")
print(f"  Protocol: Open-Set (train on bona fide only)")
print(f"  Device:   {cfg.DEVICE}")
print(f"\n  Ensemble: K={eval_results['optimal_k']} clusters, "
      f"PCA={eval_results['pca_components']} components, "
      f"contamination={eval_results['optimal_contamination'][0]:.2f} (paper value)")

print(f"\n  {'Metric':<10} {'Ours':>10} {'Paper':>10} {'Delta':>10}")
print(f"  {'-'*44}")
paper = {'ACER': 0.160, 'BPCER': 0.138, 'APCER': 0.183,
         'HTER': 0.041, 'EER':   0.034}
for metric in ['ACER', 'BPCER', 'APCER', 'HTER', 'EER']:
    ours  = overall[metric]
    ref   = paper[metric]
    delta = ours - ref
    sign  = '+' if delta > 0 else ''
    print(f"  {metric:<10} {ours*100:>9.3f}% {ref*100:>9.1f}% {sign}{delta*100:>8.3f}%")

print(f"  {'AUC-ROC':<10} {overall['AUC_ROC']:>10.4f}")

print(f"\n  Per Attack Type:")
print(f"  {'Type':<24} {'APCER':>8} {'AUC':>8} {'EER':>8} {'Set':>10}")
print(f"  {'-'*60}")
for name, m in eval_results['test_metrics']['per_attack'].items():
    set_type = "Known" if m['known_attack'] else "UNKNOWN"
    print(f"  {name:<24} {m['APCER']*100:>7.2f}% {m['AUC_ROC']:>7.4f} "
          f"{m['EER']*100:>7.2f}%  {set_type:>10}")

print(f"\n{'='*70}")
print(f"  Project Complete!")
print(f"  All results saved to: {cfg.RESULTS_DIR}")
print(f"{'='*70}")

