"""
Dataset and DataLoader for RECOD-MPAD.

Handles:
- Scanning the dataset directory structure
- Building a manifest (file list with metadata)
- User-disjoint train/val/test splitting
- Frame sampling (every Nth frame)
- Face preprocessing (detection, crop, resize)
- On-the-fly 2D-DWT decomposition
- PyTorch Dataset with lazy loading
"""

import os
import re
import csv
import glob
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from . import config as cfg
from .dwt import dwt2d_numpy


# ============================================================
# MANIFEST BUILDING
# ============================================================

def build_manifest(dataset_dir=None, output_path=None, frame_sample_rate=None):
    """
    Scan the RECOD-MPAD dataset and build a CSV manifest of all frames.

    Manifest columns:
        filepath, category, device, user_id, session, label, frame_num, split

    Args:
        dataset_dir: Path to 2020-plosone-recod-mpad root.
        output_path: Where to save the manifest CSV.
        frame_sample_rate: Use every Nth frame (default from config).

    Returns:
        List of dicts (manifest rows).
    """
    if dataset_dir is None:
        dataset_dir = cfg.DATASET_DIR
    if output_path is None:
        output_path = cfg.MANIFEST_PATH
    if frame_sample_rate is None:
        frame_sample_rate = cfg.FRAME_SAMPLE_RATE

    manifest = []

    # ---- REAL (bona fide) images ----
    real_dir = os.path.join(dataset_dir, "real")
    if os.path.exists(real_dir):
        for device in cfg.DEVICES:
            device_dir = os.path.join(real_dir, device)
            if not os.path.exists(device_dir):
                continue
            for user_folder in sorted(os.listdir(device_dir)):
                user_match = re.search(r'user_(\d+)', user_folder)
                if not user_match:
                    continue
                user_id = int(user_match.group(1))
                user_path = os.path.join(device_dir, user_folder)

                for sequence_folder in sorted(os.listdir(user_path)):
                    seq_path = os.path.join(user_path, sequence_folder)
                    if not os.path.isdir(seq_path):
                        continue

                    frames = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
                    for i, fpath in enumerate(frames):
                        if i % frame_sample_rate != 0:
                            continue

                        # Parse filename: device_session_user_label_frame.jpg
                        fname = os.path.splitext(os.path.basename(fpath))[0]
                        parts = fname.split('_')

                        split = _get_split(user_id)
                        manifest.append({
                            'filepath': fpath,
                            'category': 'real',
                            'device': device,
                            'user_id': user_id,
                            'session': int(parts[1]) if len(parts) > 1 else 0,
                            'label': 0,
                            'frame_num': int(parts[-1]) if parts[-1].isdigit() else i,
                            'split': split,
                        })

    # ---- ATTACK images ----
    attack_folders = {
        'attack_print1': 1,
        'attack_print2': 2,
        'attack_cce': 3,
        'attack_hp': 4,
    }

    for attack_folder, attack_label in attack_folders.items():
        attack_dir = os.path.join(dataset_dir, attack_folder)
        if not os.path.exists(attack_dir):
            continue

        # Navigate the nested structure to find device/user folders
        _scan_attack_dir(attack_dir, attack_folder, attack_label,
                         frame_sample_rate, manifest)

    # Assign splits
    print(f"[Manifest] Total frames: {len(manifest)}")
    for split_name in ['train', 'val', 'test']:
        count = sum(1 for m in manifest if m['split'] == split_name)
        print(f"  {split_name}: {count}")
    for label in range(5):
        count = sum(1 for m in manifest if m['label'] == label)
        print(f"  Label {label} ({cfg.ATTACK_NAMES[label]}): {count}")

    # Save CSV
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'filepath', 'category', 'device', 'user_id',
                'session', 'label', 'frame_num', 'split'
            ])
            writer.writeheader()
            writer.writerows(manifest)
        print(f"[Manifest] Saved to {output_path}")

    return manifest


def _scan_attack_dir(base_dir, category, label, sample_rate, manifest):
    """Recursively scan attack directories to find user image folders."""
    # Try different structures the dataset might have
    for device in cfg.DEVICES:
        # Structure: attack_dir/device/user_XX/...
        # or: attack_dir/attack/typeN/device/user_XX/...
        # We search recursively for user_XX folders
        for root, dirs, files in os.walk(base_dir):
            # Check if this looks like a user folder
            user_match = re.search(r'user_(\d+)', os.path.basename(root))
            if not user_match:
                continue

            user_id = int(user_match.group(1))

            # Detect device from path
            detected_device = None
            for d in cfg.DEVICES:
                if d in root:
                    detected_device = d
                    break

            # Find sequences or direct images
            jpg_files = sorted([f for f in files if f.lower().endswith('.jpg')])

            if jpg_files:
                # Images directly in user folder or sequence subfolder
                for i, fname in enumerate(jpg_files):
                    if i % sample_rate != 0:
                        continue

                    fpath = os.path.join(root, fname)
                    name_parts = os.path.splitext(fname)[0].split('_')

                    split = _get_split(user_id)
                    manifest.append({
                        'filepath': fpath,
                        'category': category,
                        'device': detected_device or 'unknown',
                        'user_id': user_id,
                        'session': int(name_parts[1]) if len(name_parts) > 1 else 0,
                        'label': label,
                        'frame_num': int(name_parts[-1]) if name_parts[-1].isdigit() else i,
                        'split': split,
                    })
            else:
                # Check sub-sequence folders
                for seq_dir in sorted(dirs):
                    seq_path = os.path.join(root, seq_dir)
                    seq_files = sorted(glob.glob(os.path.join(seq_path, "*.jpg")))
                    for i, fpath in enumerate(seq_files):
                        if i % sample_rate != 0:
                            continue

                        fname = os.path.splitext(os.path.basename(fpath))[0]
                        parts = fname.split('_')

                        split = _get_split(user_id)
                        manifest.append({
                            'filepath': fpath,
                            'category': category,
                            'device': detected_device or 'unknown',
                            'user_id': user_id,
                            'session': int(parts[1]) if len(parts) > 1 else 0,
                            'label': label,
                            'frame_num': int(parts[-1]) if parts[-1].isdigit() else i,
                            'split': split,
                        })


def _get_split(user_id):
    """Assign train/val/test split based on user ID."""
    if user_id in cfg.TRAIN_USERS:
        return 'train'
    elif user_id in cfg.VAL_USERS:
        return 'val'
    elif user_id in cfg.TEST_USERS:
        return 'test'
    else:
        return 'train'  # fallback


def load_manifest(manifest_path=None):
    """Load manifest from CSV file."""
    if manifest_path is None:
        manifest_path = cfg.MANIFEST_PATH

    manifest = []
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['user_id'] = int(row['user_id'])
            row['session'] = int(row['session'])
            row['label'] = int(row['label'])
            row['frame_num'] = int(row['frame_num'])
            manifest.append(row)

    return manifest


# ============================================================
# PYTORCH DATASET
# ============================================================

class FacePADDataset(Dataset):
    """
    PyTorch Dataset for face presentation attack detection.

    Each sample returns:
        - dwt_concat: (12, 64, 64) concatenated DWT bands
        - hh_band: (3, 64, 64) original HH band (for spectral loss)
        - label: int (0=real, 1-4=attack types)
        - metadata: dict with user_id, device, category
    """

    def __init__(self, manifest, split='train', img_size=None,
                 transform=None, bona_fide_only=False):
        """
        Args:
            manifest: List of dicts from build_manifest or load_manifest.
            split: 'train', 'val', or 'test'.
            img_size: Resize target (default from config).
            transform: Optional torchvision transforms.
            bona_fide_only: If True, only include label=0 (for training).
        """
        self.img_size = img_size or cfg.IMG_SIZE
        self.transform = transform

        # Filter by split
        self.samples = [m for m in manifest if m['split'] == split]

        # For training: only bona fide images
        if bona_fide_only:
            self.samples = [m for m in self.samples if m['label'] == 0]

        print(f"[Dataset] Split='{split}', bona_fide_only={bona_fide_only}, "
              f"samples={len(self.samples)}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        filepath = sample['filepath']
        label = sample['label']

        # Load image
        img = cv2.imread(filepath)
        if img is None:
            # Fallback: return a black image
            img = np.zeros((self.img_size, self.img_size, 3), dtype=np.uint8)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Resize to target size
        img = cv2.resize(img, (self.img_size, self.img_size),
                         interpolation=cv2.INTER_AREA)

        # Normalize to [0, 1]
        img_float = img.astype(np.float32) / 255.0

        # Apply 2D-DWT
        bands = dwt2d_numpy(img_float, wavelet='haar')

        # Extract HH band separately (for spectral loss)
        hh_band = bands['HH']  # (64, 64, 3)

        # Concatenate all bands: (64, 64, 12)
        dwt_concat = np.concatenate(
            [bands['LL'], bands['LH'], bands['HL'], bands['HH']], axis=-1
        )

        # Convert to tensors (channels first)
        dwt_concat = torch.from_numpy(
            np.transpose(dwt_concat, (2, 0, 1))  # (12, 64, 64)
        ).float()

        hh_band = torch.from_numpy(
            np.transpose(hh_band, (2, 0, 1))  # (3, 64, 64)
        ).float()

        # Apply additional transforms if provided
        if self.transform:
            dwt_concat = self.transform(dwt_concat)

        metadata = {
            'user_id': sample['user_id'],
            'device': sample['device'],
            'category': sample['category'],
            'filepath': filepath,
        }

        return dwt_concat, hh_band, label, metadata


# ============================================================
# DATA AUGMENTATION
# ============================================================

def get_train_transform():
    """Data augmentation transforms for training."""
    return transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
    ])


def get_eval_transform():
    """No augmentation for evaluation."""
    return None


# ============================================================
# DATALOADER FACTORY
# ============================================================

def create_dataloaders(manifest, batch_size=None, num_workers=None,
                       train_with_attacks=None):
    """
    Create train, validation, and test DataLoaders.

    Training loader: bona fide only OR bona fide + known attacks (for repulsion loss)
    Val loader: bona fide + known attacks (for threshold tuning)
    Test loader: bona fide + all attacks (for open-set evaluation)

    Returns:
        train_loader, val_loader, test_loader
    """
    if batch_size is None:
        batch_size = cfg.BATCH_SIZE
    if num_workers is None:
        num_workers = cfg.NUM_WORKERS
    if train_with_attacks is None:
        train_with_attacks = getattr(cfg, 'TRAIN_WITH_ATTACKS', False)

    # Training: bona fide only OR bona fide + known attacks
    # When train_with_attacks=True we include the known attack type (print1)
    # but filter out unknown attacks (print2, cce, hp)
    if train_with_attacks:
        # Include bona fide (0) and known attack (1 = print1)
        raw_manifest = [m for m in manifest
                        if m['split'] == 'train' and m['label'] in [0, 1]]
        # Use a custom filtered dataset
        train_dataset = FacePADDataset(
            raw_manifest, split='train',
            transform=get_train_transform(),
            bona_fide_only=False,
        )
    else:
        train_dataset = FacePADDataset(
            manifest, split='train',
            transform=get_train_transform(),
            bona_fide_only=True,
        )

    # Validation: bona fide + known attacks (for threshold tuning)
    val_dataset = FacePADDataset(
        manifest, split='val',
        transform=get_eval_transform(),
        bona_fide_only=False,  # Include known attacks for threshold tuning
    )

    # Test: bona fide + ALL attacks (open-set evaluation)
    test_dataset = FacePADDataset(
        manifest, split='test',
        transform=get_eval_transform(),
        bona_fide_only=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(num_workers > 0),  # Keep workers alive between epochs
        prefetch_factor=4 if num_workers > 0 else None,  # Pre-load 4 batches ahead
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=cfg.PIN_MEMORY,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    return train_loader, val_loader, test_loader
