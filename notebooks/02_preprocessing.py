# %% [markdown]
# # Notebook 2: Preprocessing & Manifest Building
# **Autoencoders for Open-Set Presentation Attack Detection**
# 
# This notebook:
# - Builds a manifest CSV of all dataset images
# - Applies user-disjoint train/val/test splitting
# - Samples every 5th frame to reduce dataset size
# - Verifies the data pipeline with a test batch

# %% [markdown]
# ## 1. Setup

# %%
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config as cfg
from src.dataset import build_manifest, load_manifest, FacePADDataset, create_dataloaders
from src.dwt import dwt2d_numpy, visualize_dwt_bands
from src.utils import set_seed

set_seed()
cfg.print_config()

# %% [markdown]
# ## 2. Build Manifest
# Scans the entire dataset and creates a CSV with metadata for each frame.

# %%
manifest = build_manifest(
    dataset_dir=cfg.DATASET_DIR,
    output_path=cfg.MANIFEST_PATH,
    frame_sample_rate=cfg.FRAME_SAMPLE_RATE,
)

print(f"\nManifest saved to: {cfg.MANIFEST_PATH}")
print(f"Total entries: {len(manifest)}")

# %% [markdown]
# ## 3. Analyze Manifest

# %%
# Load as DataFrame for analysis
df = pd.DataFrame(manifest)
print("\n--- Split Distribution ---")
print(df.groupby(['split', 'label']).size().unstack(fill_value=0))

print("\n--- Device Distribution ---")
print(df.groupby(['device', 'label']).size().unstack(fill_value=0))

print("\n--- Category × Split ---")
print(df.groupby(['category', 'split']).size().unstack(fill_value=0))

# %%
# Verify open-set protocol
print("\n--- Open-Set Protocol Verification ---")
train_labels = df[df['split'] == 'train']['label'].unique()
val_labels = df[df['split'] == 'val']['label'].unique()
test_labels = df[df['split'] == 'test']['label'].unique()

print(f"  Train labels: {sorted(train_labels)}")
print(f"  Val labels:   {sorted(val_labels)}")
print(f"  Test labels:  {sorted(test_labels)}")

train_users = sorted(df[df['split'] == 'train']['user_id'].unique())
val_users = sorted(df[df['split'] == 'val']['user_id'].unique())
test_users = sorted(df[df['split'] == 'test']['user_id'].unique())

print(f"\n  Train users: {train_users[0]}-{train_users[-1]} ({len(train_users)} users)")
print(f"  Val users:   {val_users[0]}-{val_users[-1]} ({len(val_users)} users)")
print(f"  Test users:  {test_users[0]}-{test_users[-1]} ({len(test_users)} users)")

# Verify no user overlap
assert len(set(train_users) & set(val_users)) == 0, "User overlap: train/val!"
assert len(set(train_users) & set(test_users)) == 0, "User overlap: train/test!"
assert len(set(val_users) & set(test_users)) == 0, "User overlap: val/test!"
print("\n  ✓ No user overlap between splits (user-disjoint)")

# %% [markdown]
# ## 4. Test Data Pipeline

# %%
# Create a small test dataset to verify the pipeline
print("\nTesting data pipeline...")
train_loader, val_loader, test_loader = create_dataloaders(manifest)

print(f"\nTrain batches: {len(train_loader)}")
print(f"Val batches: {len(val_loader)}")
print(f"Test batches: {len(test_loader)}")

# %%
# Grab one batch and inspect
batch = next(iter(train_loader))
dwt_concat, hh_band, labels, metadata = batch

print(f"\nBatch shapes:")
print(f"  DWT concat: {dwt_concat.shape}")   # Expected: (16, 12, 64, 64)
print(f"  HH band:    {hh_band.shape}")       # Expected: (16, 3, 64, 64)
print(f"  Labels:     {labels.shape}")         # Expected: (16,)
print(f"  Labels (values): {labels.tolist()}")

print(f"\nValue ranges:")
print(f"  DWT concat: [{dwt_concat.min():.4f}, {dwt_concat.max():.4f}]")
print(f"  HH band:    [{hh_band.min():.4f}, {hh_band.max():.4f}]")

# %%
# Visualize a batch sample
fig, axes = plt.subplots(2, 4, figsize=(16, 8))

for i in range(min(4, dwt_concat.size(0))):
    # Show LL band (first 3 channels of dwt_concat)
    ll = dwt_concat[i, :3].permute(1, 2, 0).numpy()
    ll = (ll - ll.min()) / (ll.max() - ll.min() + 1e-8)
    axes[0, i].imshow(ll)
    axes[0, i].set_title(f'LL Band (label={labels[i].item()})')
    axes[0, i].axis('off')
    
    # Show HH band
    hh = hh_band[i].permute(1, 2, 0).numpy()
    hh = (hh - hh.min()) / (hh.max() - hh.min() + 1e-8)
    axes[1, i].imshow(hh)
    axes[1, i].set_title(f'HH Band')
    axes[1, i].axis('off')

plt.suptitle('Batch Sample: LL vs HH Bands', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'batch_sample.png'), dpi=150)
plt.show()

print("\n✓ Preprocessing pipeline verified. Proceed to Notebook 03 for training.")
