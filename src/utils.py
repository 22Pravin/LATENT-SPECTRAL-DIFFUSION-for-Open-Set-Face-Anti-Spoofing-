"""
Utility functions for visualization, reproducibility, and model inspection.
"""

import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.manifold import TSNE

from . import config as cfg


def set_seed(seed=None):
    """Set random seed for reproducibility."""
    if seed is None:
        seed = cfg.SEED
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False  # Allow cuDNN to pick fastest algo
    torch.backends.cudnn.benchmark = True       # Auto-tune conv kernels for fixed input shapes (BIG speedup)
    print(f"  Random seed set to {seed}")


def check_gpu():
    """Print GPU information."""
    print("\n  GPU Information:")
    if torch.cuda.is_available():
        print(f"    CUDA available: True")
        print(f"    Device name: {torch.cuda.get_device_name(0)}")
        print(f"    VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print(f"    CUDA version: {torch.version.cuda}")
    else:
        print(f"    CUDA available: False — training will be on CPU (slow)")
    print(f"    PyTorch version: {torch.__version__}")


# ============================================================
# VISUALIZATION
# ============================================================

def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Total loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    if any(v > 0 for v in history['val_loss']):
        axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_title('Total Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Latent MSE
    axes[1].plot(epochs, history['train_latent_mse'], 'b-', label='Train', linewidth=2)
    if any(v > 0 for v in history['val_latent_mse']):
        axes[1].plot(epochs, history['val_latent_mse'], 'r-', label='Val', linewidth=2)
    axes[1].set_title('Latent MSE Loss', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MSE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Spectral L1
    axes[2].plot(epochs, history['train_spectral_l1'], 'b-', label='Train', linewidth=2)
    if any(v > 0 for v in history['val_spectral_l1']):
        axes[2].plot(epochs, history['val_spectral_l1'], 'r-', label='Val', linewidth=2)
    axes[2].set_title('Spectral L1 Loss', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('L1')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Training curves saved to {save_path}")
    plt.close()
    return fig


def plot_score_distributions(scores, threshold=None, save_path=None):
    """
    Plot score distributions for bona fide vs. each attack type.
    """
    labels = scores['labels']
    s_final = scores['s_final']

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    # Bona fide
    bf_mask = (labels == 0)
    ax.hist(s_final[bf_mask], bins=50, alpha=0.6, label='Bona Fide',
            color='#2ecc71', density=True, edgecolor='white')

    # Each attack type
    colors = ['#e74c3c', '#e67e22', '#9b59b6', '#3498db']
    for i, (attack_label, color) in enumerate(zip([1, 2, 3, 4], colors)):
        mask = (labels == attack_label)
        if mask.sum() > 0:
            name = cfg.ATTACK_NAMES[attack_label]
            known = "★" if attack_label in cfg.KNOWN_ATTACK_TYPES else "○"
            ax.hist(s_final[mask], bins=50, alpha=0.5,
                    label=f'{known} {name}', color=color, density=True,
                    edgecolor='white')

    if threshold is not None:
        ax.axvline(x=threshold, color='black', linestyle='--', linewidth=2,
                   label=f'Threshold τ={threshold:.4f}')

    ax.set_title('Score Distribution: Bona Fide vs. Attacks', fontsize=14,
                 fontweight='bold')
    ax.set_xlabel('Anomaly Score (S_final)', fontsize=12)
    ax.set_ylabel('Density', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.figtext(0.5, -0.02, '★ = Known attack (seen during val)  |  ○ = Unknown attack (open-set)',
                ha='center', fontsize=9, style='italic')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Score distributions saved to {save_path}")
    plt.close()
    return fig


def plot_roc_curves(scores, save_path=None):
    """
    Plot ROC curves: overall + per attack type.
    """
    labels = scores['labels']
    s_final = scores['s_final']
    labels_binary = (labels > 0).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Overall ROC
    fpr, tpr, _ = roc_curve(labels_binary, s_final)
    auc_score = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, 'b-', linewidth=2,
                 label=f'Overall (AUC = {auc_score:.4f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[0].set_title('Overall ROC Curve', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Per-attack ROC
    colors = ['#e74c3c', '#e67e22', '#9b59b6', '#3498db']
    bf_mask = (labels == 0)

    for attack_label, color in zip([1, 2, 3, 4], colors):
        mask = (labels == attack_label)
        if mask.sum() == 0:
            continue

        combined = bf_mask | mask
        y_true = (labels[combined] > 0).astype(int)
        y_score = s_final[combined]

        fpr_t, tpr_t, _ = roc_curve(y_true, y_score)
        auc_t = auc(fpr_t, tpr_t)

        name = cfg.ATTACK_NAMES[attack_label]
        known = "★" if attack_label in cfg.KNOWN_ATTACK_TYPES else "○"
        axes[1].plot(fpr_t, tpr_t, color=color, linewidth=2,
                     label=f'{known} {name} (AUC={auc_t:.4f})')

    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.3)
    axes[1].set_title('Per-Attack ROC Curves', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ROC curves saved to {save_path}")
    plt.close()
    return fig


def plot_dwt_visualization(image, bands_vis, save_path=None):
    """
    Visualize original image alongside its DWT sub-bands.

    Args:
        image: (H, W, 3) RGB image, uint8
        bands_vis: dict with 'LL', 'LH', 'HL', 'HH' as uint8 images
    """
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))

    titles = ['Original', 'LL (Approx)', 'LH (Horiz)', 'HL (Vert)', 'HH (Diag)']
    images = [image, bands_vis['LL'], bands_vis['LH'], bands_vis['HL'], bands_vis['HH']]

    for ax, title, img in zip(axes, titles, images):
        ax.imshow(img)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.axis('off')

    plt.suptitle('2D Discrete Wavelet Transform (Haar)', fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def plot_latent_tsne(latent_vectors, labels, save_path=None):
    """
    t-SNE visualization of the latent space.

    Args:
        latent_vectors: (N, latent_dim) numpy array
        labels: (N,) integer labels
    """
    print("  Computing t-SNE embedding (this may take a minute)...")
    tsne = TSNE(n_components=2, random_state=cfg.SEED, perplexity=30)
    embedded = tsne.fit_transform(latent_vectors)

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    colors = ['#2ecc71', '#e74c3c', '#e67e22', '#9b59b6', '#3498db']
    for label in range(5):
        mask = (labels == label)
        if mask.sum() == 0:
            continue
        name = cfg.ATTACK_NAMES[label]
        ax.scatter(embedded[mask, 0], embedded[mask, 1],
                   c=colors[label], label=name, alpha=0.6, s=20)

    ax.set_title('Latent Space t-SNE Visualization', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  t-SNE plot saved to {save_path}")
    plt.close()
    return fig


def plot_hh_reconstruction(hh_original, hh_recon, labels, n_samples=8, save_path=None):
    """
    Visualize original vs. reconstructed HH bands.

    Shows how attacks have visibly different reconstruction quality.
    """
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 2.5, 5))

    for i in range(min(n_samples, len(hh_original))):
        # Original HH
        orig = hh_original[i].transpose(1, 2, 0)  # (64,64,3)
        orig = (orig - orig.min()) / (orig.max() - orig.min() + 1e-8)
        axes[0, i].imshow(orig)
        name = cfg.ATTACK_NAMES.get(labels[i], '?')
        axes[0, i].set_title(f'{name}', fontsize=8)
        axes[0, i].axis('off')

        # Reconstructed HH
        recon = hh_recon[i].transpose(1, 2, 0)
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        axes[1, i].imshow(recon)
        axes[1, i].axis('off')

    axes[0, 0].set_ylabel('Original HH', fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel('Reconstructed HH', fontsize=10, fontweight='bold')
    plt.suptitle('HH Band: Original vs. Reconstructed', fontsize=14, fontweight='bold')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    return fig


def model_summary(model):
    """Print a summary of the model architecture."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model Summary:")
    print(f"    Total parameters:     {total_params:>12,}")
    print(f"    Trainable parameters: {trainable_params:>12,}")
    print(f"    Model size (MB):      {total_params * 4 / 1024**2:>12.2f} (FP32)")
    print(f"    Model size (MB):      {total_params * 2 / 1024**2:>12.2f} (FP16)")

    print(f"\n  Components:")
    for name, module in model.named_children():
        params = sum(p.numel() for p in module.parameters())
        print(f"    {name:<25} {params:>10,} params")

    return total_params, trainable_params
