"""
Configuration file for Autoencoders for Open-Set Presentation Attack Detection.
All hyperparameters, paths, and settings are centralized here.
"""

import os
import torch

# ============================================================
# PATHS
# ============================================================
# Base project directory
PROJECT_DIR = r"c:\BS_Shivang"

# Dataset root (RECOD-MPAD)
DATASET_DIR = os.path.join(PROJECT_DIR, "2020-plosone-recod-mpad")

# Output directories
PREPROCESSED_DIR = os.path.join(PROJECT_DIR, "preprocessed")
CHECKPOINT_DIR = os.path.join(PROJECT_DIR, "checkpoints")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")
MANIFEST_PATH = os.path.join(PROJECT_DIR, "manifest.csv")

# Create output directories
for d in [PREPROCESSED_DIR, CHECKPOINT_DIR, RESULTS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# DATASET STRUCTURE
# ============================================================
# Attack type mapping
ATTACK_TYPES = {
    "real": 0,          # Bona fide / genuine
    "attack_print1": 1, # Printout attack - recaptured indoors
    "attack_print2": 2, # Printout attack - recaptured outdoors
    "attack_cce": 3,    # Screen attack - CCE TV (large display)
    "attack_hp": 4,     # Screen attack - HP monitor (medium display)
}

ATTACK_NAMES = {
    0: "Bona Fide",
    1: "Print (Indoor)",
    2: "Print (Outdoor)",
    3: "Screen (CCE TV)",
    4: "Screen (HP Monitor)",
}

# Devices in the dataset
DEVICES = ["motog5", "xt1572"]

# Total number of users
NUM_USERS = 45

# ============================================================
# OPEN-SET PROTOCOL (Protocol A)
# ============================================================
# Train: Bona fide only (users 1-30)
# Validation: Bona fide + print1 (users 31-37) — for threshold tuning
# Test: Bona fide + ALL attack types (users 38-45)

TRAIN_USERS = list(range(1, 31))    # Users 1-30 (30 users)
VAL_USERS = list(range(31, 38))     # Users 31-37 (7 users)
TEST_USERS = list(range(38, 46))    # Users 38-45 (8 users)

# Known attack type used for validation threshold tuning
KNOWN_ATTACK_TYPES = [1]  # print1 only
# Unknown attack types (open-set, test only)
UNKNOWN_ATTACK_TYPES = [2, 3, 4]  # print2, cce, hp

# ============================================================
# PREPROCESSING
# ============================================================
IMG_SIZE = 128               # Input image size (128x128)
DWT_SIZE = IMG_SIZE // 2     # Wavelet sub-band size (64x64)
FRAME_SAMPLE_RATE = 5        # Use every 5th frame
FACE_MARGIN = 20             # Margin around detected face (pixels)

# ============================================================
# MODEL ARCHITECTURE
# ============================================================
LATENT_DIM = 128             # Latent space dimensionality (Bottlenecked to prevent identity memorization)
ENCODER_CHANNELS = [32, 64, 128, 256]  # Encoder channel progression
NOISE_EMBED_DIM = 64         # Noise level embedding dimension
DENOISER_HIDDEN = 1024       # Denoiser hidden layer width

# ============================================================
# DIFFUSION PROCESS
# ============================================================
NOISE_LEVEL = 0.5            # Default noise level for forward diffusion
NOISE_LEVEL_MIN = 0.1        # Min noise for training augmentation
NOISE_LEVEL_MAX = 1.2        # Max noise for training augmentation (Forced higher to disrupt screen patterns)

# ============================================================
# TRAINING
# ============================================================
BATCH_SIZE = 32              # Batch size (RTX 4090 has 24GB VRAM, can handle larger batches)
NUM_EPOCHS = 80              # Total training epochs (Phase-1: 20 frozen, Phase-2: 60 fine-tuned)
LEARNING_RATE = 3e-4         # AdamW learning rate (head + denoiser only)
WEIGHT_DECAY = 1e-4          # AdamW weight decay
ALPHA = 0.3                  # Weight for latent MSE loss (secondary signal)
BETA = 1.5                   # Weight for spectral L1 loss (primary anomaly signal)
GAMMA = 0.0                  # Center Loss weight (disabled — causes collapse)
DELTA = 0.1                  # Compact loss (Deep SVDD): pulls bona-fide z0 to tight cluster
EPSILON = 0.1                # Repulsion loss weight: pushes known attack (print1) z0 away from centroid
REPULSION_MARGIN = 50.0      # Margin (squared L2) for repulsion hinge loss.
                             # In 512-dim space, untrained dist^2 ~500; after compact
                             # training bona-fide dist^2 ~10-50. Margin=50 ensures
                             # attacks are pushed well beyond the BF cluster boundary.
EARLY_STOP_PATIENCE = 20     # Stop if no improvement for N epochs
NUM_WORKERS = 0              # Parallel data loading workers (RTX 4090 needs fed faster)
USE_AMP = True               # Mixed precision training (FP16)
PIN_MEMORY = True            # Faster GPU transfer
FREEZE_BACKBONE = True       # Freeze ResNet layers 1-4 (only train head+denoiser+decoder)
TRAIN_WITH_ATTACKS = True    # Include known attack (print1) for repulsion margin ONLY.
                             # Open-set: unknown attacks (print2, cce, hp) still NEVER seen.

# ============================================================
# EVALUATION / SCORING
# ============================================================
LAMBDA_FUSION = 0.5          # Weighted fusion: lambda*S_iso + (1-lambda)*S_latent_norm
THRESHOLD_TAU = None         # Set automatically from val BPCER=10% operating point
CONTAMINATION = 0.14         # IF contamination: stricter bona-fide boundary (matches paper value)

# Score fusion weights (used in evaluate.py)
# W_SPEC is now elevated: Spectral L1 is the primary training signal (BETA=2.0).
# W_ISO remains high: Isolation Forest is the backbone of the AOPAD ensemble.
# Weights must sum to 1.0.
W_ISO      = 0.20   # Isolation Forest weight  (reliable one-class signal)
W_LMSE     = 0.05   # Latent MSE weight         (secondary signal, kept low)
W_SPEC     = 0.60   # Spectral L1 weight        (PRIMARY signal, elevated from 0.40)
W_MAHA     = 0.15   # Mahalanobis distance      (stable distance, slightly reduced)

# Threshold calibration
BPCER_TARGET = 0.10          # Calibrate threshold so BPCER <= 10% on val set (matches paper operating point)

# ============================================================
# DEVICE
# ============================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# RANDOM SEED
# ============================================================
SEED = 42


def print_config():
    """Print current configuration for logging."""
    print("=" * 60)
    print("CONFIGURATION")
    print("=" * 60)
    print(f"  Device:            {DEVICE}")
    print(f"  Image Size:        {IMG_SIZE}x{IMG_SIZE}")
    print(f"  DWT Sub-band Size: {DWT_SIZE}x{DWT_SIZE}")
    print(f"  Latent Dim:        {LATENT_DIM}")
    print(f"  Batch Size:        {BATCH_SIZE}")
    print(f"  Epochs:            {NUM_EPOCHS}")
    print(f"  Learning Rate:     {LEARNING_RATE}")
    print(f"  Mixed Precision:   {USE_AMP}")
    print(f"  Loss Weights:      alpha={ALPHA}, beta={BETA}, gamma={GAMMA}")
    print(f"  Fusion Weight:     lambda={LAMBDA_FUSION}")
    print(f"  Anomaly Weights:   w_iso={W_ISO}, w_lmse={W_LMSE}, w_spec={W_SPEC}, w_maha={W_MAHA}")
    print(f"  IF Contamination:  {CONTAMINATION} (paper value)")
    print(f"  Train Users:       {len(TRAIN_USERS)} (IDs {TRAIN_USERS[0]}-{TRAIN_USERS[-1]})")
    print(f"  Val Users:         {len(VAL_USERS)} (IDs {VAL_USERS[0]}-{VAL_USERS[-1]})")
    print(f"  Test Users:        {len(TEST_USERS)} (IDs {TEST_USERS[0]}-{TEST_USERS[-1]})")
    print(f"  Known Attacks:     {KNOWN_ATTACK_TYPES}")
    print(f"  Unknown Attacks:   {UNKNOWN_ATTACK_TYPES}")
    print("=" * 60)
