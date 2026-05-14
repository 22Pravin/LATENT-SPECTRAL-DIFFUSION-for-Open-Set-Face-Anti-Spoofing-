# %% [markdown]
# # Notebook 1: Data Exploration & Visualization
# **Autoencoders for Open-Set Presentation Attack Detection**
# 
# This notebook explores the RECOD-MPAD dataset:
# - Dataset statistics (images per class, device, session)
# - Sample visualization (real vs. attack types)
# - Image resolution and quality analysis
# - 2D-DWT decomposition visualization

# %% [markdown]
# ## 1. Setup & Imports

# %%
import os
import sys
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
from collections import Counter

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src import config as cfg
from src.dwt import dwt2d_numpy, visualize_dwt_bands
from src.utils import set_seed, check_gpu

set_seed()
check_gpu()
cfg.print_config()

# %% [markdown]
# ## 2. Scan Dataset & Count Images

# %%
def count_images_in_dir(base_dir):
    """Recursively count JPG files."""
    count = 0
    for root, dirs, files in os.walk(base_dir):
        count += sum(1 for f in files if f.lower().endswith('.jpg'))
    return count

print("Dataset: RECOD-MPAD")
print(f"Location: {cfg.DATASET_DIR}\n")

categories = {
    'real': os.path.join(cfg.DATASET_DIR, 'real'),
    'attack_print1': os.path.join(cfg.DATASET_DIR, 'attack_print1'),
    'attack_print2': os.path.join(cfg.DATASET_DIR, 'attack_print2'),
    'attack_cce': os.path.join(cfg.DATASET_DIR, 'attack_cce'),
    'attack_hp': os.path.join(cfg.DATASET_DIR, 'attack_hp'),
}

stats = {}
for name, path in categories.items():
    if os.path.exists(path):
        count = count_images_in_dir(path)
        stats[name] = count
        label = cfg.ATTACK_NAMES[cfg.ATTACK_TYPES[name]]
        print(f"  {label:<20} ({name}): {count:>8,} images")
    else:
        print(f"  {name}: NOT FOUND")
        stats[name] = 0

total = sum(stats.values())
print(f"\n  Total: {total:,} images")

# %%
# Bar chart of class distribution
fig, ax = plt.subplots(figsize=(10, 5))
names = [cfg.ATTACK_NAMES[cfg.ATTACK_TYPES[k]] for k in stats.keys()]
counts = list(stats.values())
colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6', '#3498db']

bars = ax.bar(names, counts, color=colors, edgecolor='white', linewidth=1.5)
ax.set_title('RECOD-MPAD: Images per Category', fontsize=14, fontweight='bold')
ax.set_ylabel('Number of Images', fontsize=12)

for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 500,
            f'{count:,}', ha='center', va='bottom', fontweight='bold')

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'class_distribution.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 3. Visualize Sample Images

# %%
def get_sample_images(base_dir, n=3):
    """Get N sample image paths from a directory."""
    all_jpgs = []
    for root, dirs, files in os.walk(base_dir):
        for f in files:
            if f.lower().endswith('.jpg'):
                all_jpgs.append(os.path.join(root, f))
                if len(all_jpgs) >= 500:
                    break
    
    if len(all_jpgs) > n:
        indices = np.linspace(0, len(all_jpgs)-1, n, dtype=int)
        return [all_jpgs[i] for i in indices]
    return all_jpgs[:n]

n_samples = 4
fig, axes = plt.subplots(5, n_samples, figsize=(n_samples*3, 15))

for row, (cat_name, cat_path) in enumerate(categories.items()):
    if not os.path.exists(cat_path):
        continue
    
    samples = get_sample_images(cat_path, n=n_samples)
    label = cfg.ATTACK_NAMES[cfg.ATTACK_TYPES[cat_name]]
    
    for col, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            axes[row, col].imshow(img)
            if col == 0:
                axes[row, col].set_ylabel(label, fontsize=10, fontweight='bold')
        axes[row, col].axis('off')

plt.suptitle('Sample Images: Real vs. Attack Types', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'sample_images.png'), dpi=150)
plt.show()

# %% [markdown]
# ## 4. Image Resolution Analysis

# %%
resolutions = []
for cat_name, cat_path in categories.items():
    if not os.path.exists(cat_path):
        continue
    samples = get_sample_images(cat_path, n=20)
    for img_path in samples:
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            resolutions.append({'category': cat_name, 'height': h, 'width': w})

res_df = pd.DataFrame(resolutions)
print("Image Resolutions:")
print(res_df.groupby('category')[['height', 'width']].describe())

# %% [markdown]
# ## 5. 2D-DWT Decomposition Visualization

# %%
# Show DWT decomposition for one real and one attack image
for cat_name in ['real', 'attack_print1']:
    cat_path = categories[cat_name]
    if not os.path.exists(cat_path):
        continue
    
    sample_path = get_sample_images(cat_path, n=1)[0]
    img = cv2.imread(sample_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
    
    # Apply DWT
    bands_vis = visualize_dwt_bands(img_resized)
    
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    
    titles = ['Original', 'LL (Approx)', 'LH (Horiz Detail)', 
              'HL (Vert Detail)', 'HH (Diag Detail)']
    images = [img_resized, bands_vis['LL'], bands_vis['LH'], 
              bands_vis['HL'], bands_vis['HH']]
    
    for ax, title, im in zip(axes, titles, images):
        ax.imshow(im)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')
    
    label = cfg.ATTACK_NAMES[cfg.ATTACK_TYPES[cat_name]]
    plt.suptitle(f'2D-DWT Decomposition — {label}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(cfg.RESULTS_DIR, f'dwt_{cat_name}.png'), dpi=150)
    plt.show()

# %% [markdown]
# ## 6. Compare HH Bands: Real vs. Attack
# The HH (diagonal detail) band captures high-frequency artifacts.
# Attacks typically show different HH patterns compared to real faces.

# %%
fig, axes = plt.subplots(2, 5, figsize=(20, 8))

for row, cat_name in enumerate(['real', 'attack_print1']):
    cat_path = categories[cat_name]
    if not os.path.exists(cat_path):
        continue
    
    samples = get_sample_images(cat_path, n=5)
    label = cfg.ATTACK_NAMES[cfg.ATTACK_TYPES[cat_name]]
    
    for col, img_path in enumerate(samples):
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (cfg.IMG_SIZE, cfg.IMG_SIZE))
        
        bands = visualize_dwt_bands(img_resized)
        axes[row, col].imshow(bands['HH'])
        axes[row, col].axis('off')
    
    axes[row, 0].set_ylabel(label, fontsize=11, fontweight='bold')

plt.suptitle('HH Band Comparison: Bona Fide vs. Print Attack', 
             fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(cfg.RESULTS_DIR, 'hh_comparison.png'), dpi=150)
plt.show()

print("\n✓ Data exploration complete. Proceed to Notebook 02 for preprocessing.")
