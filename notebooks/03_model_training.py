# %% [markdown]
# # Notebook 3: Model Training (v2 — Backbone Frozen + Compact-Repulsion Loss)
# **Autoencoders for Open-Set Presentation Attack Detection**
#
# Key changes vs. v1:
#   - ResNet-18 backbone (layers 1-4) is FROZEN — only the FC head, denoiser,
#     and spectral decoder are trained (~34% of params). This prevents the
#     powerful pretrained features from generalizing to attack images.
#   - Compact-Repulsion loss replaces the broken Center Loss. It:
#       * Pulls bona-fide z0 embeddings toward a learnable centroid (compact)
#       * Pushes known-attack z0 embeddings away (repulsion)
#   - Known-attack frames (print1) are included in the training DataLoader
#     to provide the repulsion signal.
#   - CosineAnnealingWarmRestarts scheduler for better LR coverage.

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
from src.model import build_model, DWTAutoencoder
from src.train import Trainer
from src.utils import set_seed, check_gpu, model_summary

set_seed()
check_gpu()
cfg.print_config()

# %% [markdown]
# ## 2. Load Data

# %%
manifest = load_manifest(cfg.MANIFEST_PATH)
print(f"Loaded manifest: {len(manifest)} entries")

# TRAIN_WITH_ATTACKS=True includes bona-fide + known attacks (print1)
# in the training loader so the repulsion loss has signal.
train_loader, val_loader, test_loader = create_dataloaders(manifest)

# Quick sanity check on train set composition
from collections import Counter
labels_in_train = [m['label'] for m in manifest if m['split'] == 'train'
                   and m['label'] in [0, 1]]
print(f"\nTrain set (with attacks): {Counter(labels_in_train)}")
print(f"  0=Bona Fide, 1=Print (Indoor) — only label=1 used for repulsion")

# %% [markdown]
# ## 3. Build Model

# %%
model = build_model()
model_summary(model)

# Trainable vs frozen parameters
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"\nParameter budget:")
print(f"  Total:     {total:,}")
print(f"  Trainable: {trainable:,}  ({100*trainable/total:.1f}%)")
print(f"  Frozen:    {total-trainable:,}  (ResNet backbone layers 1-4)")

# %% [markdown]
# ## 4. Train the Model

# %%
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    learning_rate=cfg.LEARNING_RATE,
    num_epochs=cfg.NUM_EPOCHS,
    device=cfg.DEVICE,
    checkpoint_dir=cfg.CHECKPOINT_DIR,
    use_amp=cfg.USE_AMP,
)

# %%
history = trainer.train()

# %% [markdown]
# ## 5. Training Curves

# %%
fig, axes = plt.subplots(1, 4, figsize=(24, 5))
epochs = range(1, len(history['train_loss']) + 1)

axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
if any(v > 0 for v in history['val_loss']):
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
axes[0].set_title('Total Loss')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(epochs, history['train_latent_mse'], 'b-', label='Latent MSE', linewidth=2)
axes[1].plot(epochs, history['train_spectral_l1'], 'g-', label='Spectral L1', linewidth=2)
axes[1].set_title('Reconstruction Losses')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(epochs, history.get('train_compact', [0]*len(epochs)), 'orange', label='Compact', linewidth=2)
axes[2].plot(epochs, history.get('train_repulsion', [0]*len(epochs)), 'red', label='Repulsion', linewidth=2)
axes[2].set_title('Compact-Repulsion Loss')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

# Reconstruction gap (positive = attacks have higher error = model is discriminating)
if 'val_recon_gap' in history and any(v != 0 for v in history['val_recon_gap']):
    axes[3].plot(epochs, history['val_recon_gap'], 'm-', linewidth=2)
    axes[3].axhline(0, color='k', linestyle='--', alpha=0.5)
    axes[3].set_title('Val Reconstruction Gap\n(attack L1 - bona fide L1)\n[positive = model discriminates!]')
    axes[3].grid(True, alpha=0.3)
else:
    axes[3].text(0.5, 0.5, 'No gap data\n(run validation)', ha='center', va='center',
                 transform=axes[3].transAxes)
    axes[3].set_title('Val Reconstruction Gap')

plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'training_curves.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 6. Verify Discrimination via Reconstruction Errors

# %%
# After training, test that attacks have meaningfully higher reconstruction errors
model_best = build_model()
ckpt = torch.load(os.path.join(cfg.CHECKPOINT_DIR, 'best_model.pth'), map_location=cfg.DEVICE)
model_best.load_state_dict(ckpt['model_state_dict'])
model_best = model_best.to(cfg.DEVICE)
model_best.eval()

results_by_class = {i: {'lmse': [], 'sl1': []} for i in range(5)}

with torch.no_grad():
    for dwt_concat, hh_band, labels, _ in test_loader:
        dwt_concat = dwt_concat.to(cfg.DEVICE)
        hh_band    = hh_band.to(cfg.DEVICE)
        nl = torch.full((dwt_concat.size(0),), cfg.NOISE_LEVEL, device=cfg.DEVICE)
        out = model_best(dwt_concat, hh_band, noise_level=nl)
        for i in range(len(labels)):
            l = labels[i].item()
            results_by_class[l]['lmse'].append(out['latent_mse'][i].item())
            results_by_class[l]['sl1'].append(out['spectral_l1'][i].item())

print("\nReconstruction Errors by Class (GOAL: attacks > bona fide):")
print(f"  {'Class':<25} {'Latent MSE':>12} {'Spectral L1':>12}")
print(f"  {'-'*51}")
bf_sl1 = np.mean(results_by_class[0]['sl1'])
for k in range(5):
    lmse = np.mean(results_by_class[k]['lmse'])
    sl1  = np.mean(results_by_class[k]['sl1'])
    gap  = sl1 - bf_sl1
    gap_str = f"  [+{gap*1000:.2f}e-3]" if gap > 0 else f"  [{gap*1000:.2f}e-3]"
    print(f"  {cfg.ATTACK_NAMES[k]:<25} {lmse:>12.6f} {sl1:>12.6f}{gap_str if k>0 else ''}")

print("\n[v] Training complete. Proceed to Notebook 04 for evaluation.")
