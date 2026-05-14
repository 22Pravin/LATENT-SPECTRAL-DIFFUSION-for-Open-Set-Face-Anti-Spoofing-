# %% [markdown]
# # Notebook 4: Open-Set Evaluation (AOPAD + Calibrated Threshold)
# **Autoencoders for Open-Set Presentation Attack Detection**
#
# Pipeline (v3 — Two-Phase Training + SVDD Centroid):
#   1. Load best trained model (two-phase: backbone frozen phase-1, layer3+4 unfrozen phase-2)
#   2. Extract z0 latents + reconstruction errors (latent_mse, spectral_l1)
#      from bona-fide training data
#   3. Fit PCA + K-Means + per-cluster Isolation Forests on bona-fide features
#      (features include reconstruction errors + LBP texture)
#   4. Score val set:
#        - Run score inversion guard (flips scores if attacks score lower than bona-fide)
#        - Calibrate threshold at BPCER <= 5% (tighter than v2)
#   5. Apply calibrated threshold on test set — no data leakage
#   6. Report APCER, BPCER, ACER, HTER, EER, AUC-ROC per attack type

# %% [markdown]
# ## 1. Setup

# %%
import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config as cfg
from src.dataset import load_manifest, create_dataloaders
from src.model import build_model
from src.evaluate import AOPADEvaluator
from src.utils import set_seed

set_seed()
cfg.print_config()

# %% [markdown]
# ## 2. Load Model & Data

# %%
model = build_model()
checkpoint = torch.load(
    os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'),
    map_location=cfg.DEVICE,
)
model.load_state_dict(checkpoint['model_state_dict'])
print(f"[v] Loaded model from epoch {checkpoint['epoch']+1} "
      f"(loss: {checkpoint['loss']:.6f})")

manifest = load_manifest()
train_loader, val_loader, test_loader = create_dataloaders(manifest)

# %% [markdown]
# ## 3. Fit AOPAD Ensemble & Run Evaluation

# %%
evaluator = AOPADEvaluator(
    model=model,
    device=cfg.DEVICE,
    noise_level=cfg.NOISE_LEVEL,
    k_max=15,
    pca_variance=0.95,
)

# Full evaluation: fits on train, calibrates on val, tests on test
results, val_scores, test_scores = evaluator.full_evaluation(
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    save_dir=cfg.RESULTS_DIR,
)

# %% [markdown]
# ## 4. Score Distributions

# %%
colors_map = {0: '#2ecc71', 1: '#e74c3c', 2: '#e67e22', 3: '#9b59b6', 4: '#3498db'}

def plot_scores(scores_dict, title, threshold=None):
    labels  = scores_dict['labels']
    s_final = scores_dict['s_final']
    fig, ax = plt.subplots(figsize=(12, 5))
    for lbl in sorted(np.unique(labels)):
        mask = labels == lbl
        name = cfg.ATTACK_NAMES[lbl]
        known = " [Known]" if lbl in cfg.KNOWN_ATTACK_TYPES else (" [Unknown]" if lbl > 0 else "")
        ax.hist(s_final[mask], bins=60, alpha=0.55,
                label=f"{name}{known}", color=colors_map[lbl], density=True)
    if threshold is not None:
        ax.axvline(threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold = {threshold:.4f}')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('Anomaly Score  (higher = more suspicious)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fn = title.split(':')[0].replace(' ', '_').lower()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, f"score_dist_{fn}.png"), dpi=150)
    plt.show()

plot_scores(val_scores,  "Validation: Score Distribution",
            threshold=evaluator.calibrated_threshold)
plot_scores(test_scores, "Test: Score Distribution (Open-Set)",
            threshold=evaluator.calibrated_threshold)

# %% [markdown]
# ## 5. ROC Curves

# %%
from sklearn.metrics import roc_curve, auc as sk_auc

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Overall ROC
labels_bin = (test_scores['labels'] > 0).astype(int)
fpr, tpr, _ = roc_curve(labels_bin, test_scores['s_final'])
roc_auc = sk_auc(fpr, tpr)
ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'Overall (AUC={roc_auc:.4f})')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax1.set_title('Overall ROC', fontweight='bold')
ax1.set_xlabel('FPR')
ax1.set_ylabel('TPR')
ax1.legend()
ax1.grid(alpha=0.3)

# Per-attack ROC
bf_mask = (test_scores['labels'] == 0)
colors_list = ['#e74c3c', '#e67e22', '#9b59b6', '#3498db']
for attack_lbl, color in zip([1, 2, 3, 4], colors_list):
    mask = (test_scores['labels'] == attack_lbl)
    if mask.sum() == 0:
        continue
    combined = bf_mask | mask
    y_true   = (test_scores['labels'][combined] > 0).astype(int)
    y_score  = test_scores['s_final'][combined]
    fpr_t, tpr_t, _ = roc_curve(y_true, y_score)
    auc_t = sk_auc(fpr_t, tpr_t)
    name  = cfg.ATTACK_NAMES[attack_lbl]
    known = "[K]" if attack_lbl in cfg.KNOWN_ATTACK_TYPES else "[U]"
    ax2.plot(fpr_t, tpr_t, color=color, linewidth=2,
             label=f'{known} {name} (AUC={auc_t:.4f})')

ax2.plot([0, 1], [0, 1], 'k--', alpha=0.3)
ax2.set_title('Per-Attack ROC', fontweight='bold')
ax2.set_xlabel('FPR')
ax2.set_ylabel('TPR')
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'roc_curves.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Results Summary Table

# %%
import pandas as pd

rows = []
for name, m in results['test_metrics']['per_attack'].items():
    rows.append({
        'Attack Type': name,
        'Known?':      'Yes' if m['known_attack'] else 'No (Open-Set)',
        'APCER (%)':   f"{m['APCER']*100:.2f}",
        'AUC-ROC':     f"{m['AUC_ROC']:.4f}",
        'EER (%)':     f"{m['EER']*100:.2f}",
        'N Samples':   m['n_samples'],
    })

results_df = pd.DataFrame(rows)
print("\n" + "=" * 80)
print("OPEN-SET PAD — AOPAD v2 EVALUATION RESULTS")
print("=" * 80)
print(results_df.to_string(index=False))

o = results['test_metrics']['overall']
print(f"\nOverall:  ACER={o['ACER']*100:.2f}%  APCER={o['APCER']*100:.2f}%  "
      f"BPCER={o['BPCER']*100:.2f}%  HTER={o['HTER']*100:.2f}%")
print(f"          EER={o['EER']*100:.2f}%   AUC-ROC={o['AUC_ROC']:.4f}  "
      f"Accuracy={o['accuracy']*100:.2f}%")

print(f"\nPaper targets:  ACER=16.0%  APCER=18.3%  BPCER=13.8%  HTER=4.1%  EER=3.4%")

print(f"\nCalibrated threshold: {results.get('calibrated_threshold', 'N/A')}")
print(f"Ensemble: K={results['optimal_k']} clusters, "
      f"PCA={results['pca_components']} components, "
      f"contamination={results['optimal_contamination'][0]:.2f}")
print(f"Score inverted by guard: {getattr(evaluator, 'score_inverted', False)}")
print("\n[v] Evaluation complete. Proceed to Notebook 05 for detailed analysis.")
