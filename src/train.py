"""
Training loop for the DWT Denoising Autoencoder.

Two-Phase Training Strategy:
  Phase 1 (epochs 0-19, backbone FROZEN):
    - Only FC head, denoiser, spectral decoder are trained.
    - After epoch 0: centroid is initialized from mean bona-fide z0.
    - Compact loss (Deep SVDD style) activates from epoch 1 onward.

  Phase 2 (epochs 20+, partial backbone UNFREEZE):
    - layer3 + layer4 + FC of ResNet are unfrozen.
    - A new optimizer param group with 10x smaller LR is added for these layers.
    - This allows fine-tuning high-level features without catastrophic forgetting.

Key changes vs v1:
  1. Centroid initialization: collected from epoch-0 bona-fide z0, then frozen.
  2. Two-phase backbone: frozen -> partial unfreeze at epoch PHASE2_EPOCH.
  3. Separate LR for backbone fine-tune (LEARNING_RATE / 10).
  4. Validation reconstruction gap tracking (attack_L1 - bf_L1 > 0 is good).
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
import json
import numpy as np

from . import config as cfg
from .model import DWTAutoencoder, build_model
from .losses import CombinedPADLoss

# Epoch at which Phase-2 (partial backbone unfreeze) begins
PHASE2_EPOCH = 20


class Trainer:
    """
    Trainer for the DWT Autoencoder-based PAD system.

    Training strategy:
      Phase 1 (epochs 0-19):
        - Freeze ResNet backbone (all layers)
        - Train FC head + denoiser + spectral decoder only
        - After epoch 0: initialize centroid from mean bona-fide z0

      Phase 2 (epoch 20+):
        - Unfreeze ResNet layer3 + layer4 (keep layer1, layer2 frozen)
        - Add new optimizer param group with LR = base_LR / 10
        - Continue compact loss with fixed centroid
    """

    def __init__(self, model, train_loader, val_loader=None,
                 learning_rate=None, num_epochs=None, device=None,
                 checkpoint_dir=None, use_amp=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device or cfg.DEVICE
        self.num_epochs = num_epochs or cfg.NUM_EPOCHS
        self.checkpoint_dir = checkpoint_dir or cfg.CHECKPOINT_DIR
        self.use_amp = use_amp if use_amp is not None else cfg.USE_AMP

        # Move model to device
        self.model = self.model.to(self.device)

        # Phase 1: freeze backbone
        if getattr(cfg, 'FREEZE_BACKBONE', False):
            self.model.encoder.freeze_backbone()

        # Loss function — compact loss (delta) + repulsion margin (epsilon)
        self.criterion = CombinedPADLoss(
            alpha=cfg.ALPHA,
            beta=cfg.BETA,
            gamma=0.0,
            delta=getattr(cfg, 'DELTA', 0.1),
            epsilon=getattr(cfg, 'EPSILON', 0.0),
            repulsion_margin=getattr(cfg, 'REPULSION_MARGIN', 10.0),
            latent_dim=cfg.LATENT_DIM,
        ).to(self.device)

        # Only optimize trainable parameters (Phase-1: head + denoiser + decoder)
        lr = learning_rate or cfg.LEARNING_RATE
        self.base_lr = lr
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        trainable_params += list(self.criterion.parameters())

        self.optimizer = optim.AdamW(
            trainable_params,
            lr=lr,
            weight_decay=cfg.WEIGHT_DECAY,
        )

        # Cosine LR with warm restarts for better convergence
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=20, T_mult=1, eta_min=1e-6
        )

        # Mixed precision scaler
        self.scaler = GradScaler('cuda', enabled=self.use_amp)

        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0

        # Phase-2 transition flag
        self._phase2_activated = False

        self.history = {
            'train_loss': [],
            'train_latent_mse': [],
            'train_spectral_l1': [],
            'train_center_loss': [],
            'train_compact': [],
            'train_repulsion': [],
            'val_loss': [],
            'val_latent_mse': [],
            'val_spectral_l1': [],
            'val_center_loss': [],
            'val_recon_gap': [],   # attack_spectral_l1 - bona_fide_spectral_l1 (want > 0)
            'lr': [],
        }

        os.makedirs(self.checkpoint_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Centroid initialization (called after epoch 0)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _initialize_centroid(self):
        """
        Collect all bona-fide z0 embeddings from the training set and use
        their mean to initialize the Deep SVDD centroid. This is called once
        after epoch 0 (before the compact loss activates at epoch 1).
        """
        print("\n  [Centroid Init] Computing mean bona-fide z0 from train set...")
        self.model.eval()
        z0_sum   = torch.zeros(cfg.LATENT_DIM, device=self.device)
        n_total  = 0

        for dwt_concat, hh_band, labels, metadata in tqdm(
                self.train_loader, desc="  [Centroid] Collecting z0"):
            dwt_concat = dwt_concat.to(self.device)
            hh_band    = hh_band.to(self.device)
            labels_dev = labels.to(self.device)

            with autocast('cuda', enabled=self.use_amp):
                out = self.model(dwt_concat, hh_band)
            # All training samples should be bona-fide, but guard anyway
            bf_mask = (labels_dev == 0)
            if bf_mask.any():
                z0_sum  += out['z0'][bf_mask].sum(dim=0).float()
                n_total += bf_mask.sum().item()

        if n_total > 0:
            mean_z0 = z0_sum / n_total
            self.criterion.initialize_center(mean_z0)
        else:
            print("  [Centroid Init] WARNING: no bona-fide samples found!")

        self.model.train()

    # ------------------------------------------------------------------
    # Phase-2 activation (called at epoch PHASE2_EPOCH)
    # ------------------------------------------------------------------

    def _activate_phase2(self):
        """
        Unfreeze ResNet layer3 + layer4 + fc and add them as a new optimizer
        param group with a 10x smaller learning rate.
        """
        print(f"\n  {'='*55}")
        print(f"  PHASE 2 ACTIVATED (epoch {PHASE2_EPOCH+1})")
        print(f"  Unfreezing ResNet layer3 + layer4 with LR = {self.base_lr/10:.1e}")
        print(f"  {'='*55}")

        self.model.encoder.unfreeze_top_layers()

        # Collect params already registered in the optimizer (to avoid duplicates)
        existing_params = set()
        for group in self.optimizer.param_groups:
            for p in group['params']:
                existing_params.add(id(p))

        # Only add newly unfrozen params that are NOT already in the optimizer
        backbone_params = [
            p for p in self.model.encoder.parameters()
            if p.requires_grad and id(p) not in existing_params
        ]

        if not backbone_params:
            print("  [Phase-2] WARNING: No new backbone params to add (all already in optimizer).")
        else:
            # Add as a separate param group with smaller LR
            self.optimizer.add_param_group({
                'params':       backbone_params,
                'lr':           self.base_lr / 10.0,
                'weight_decay': cfg.WEIGHT_DECAY,
            })

        self._phase2_activated = True
        n_trainable = sum(p.numel() for p in self.model.parameters()
                          if p.requires_grad)
        print(f"  Total trainable params after Phase-2: {n_trainable:,}\n")

    # ------------------------------------------------------------------
    # Train epoch
    # ------------------------------------------------------------------

    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        epoch_losses = {
            'total': 0, 'latent_mse': 0, 'spectral_l1': 0,
            'center_loss': 0, 'compact': 0, 'repulsion': 0
        }
        num_batches = 0

        pbar = tqdm(self.train_loader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} [Train]"
                         f" {'[Phase-2]' if self._phase2_activated else '[Phase-1]'}")

        for batch_idx, (dwt_concat, hh_band, labels, metadata) in enumerate(pbar):
            dwt_concat = dwt_concat.to(self.device, non_blocking=True)
            hh_band    = hh_band.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast('cuda', enabled=self.use_amp):
                output = self.model(dwt_concat, hh_band)
                loss, loss_dict = self.criterion(output, labels_dev)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                max_norm=1.0
            )
            self.scaler.step(self.optimizer)
            self.scaler.update()

            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0)
            num_batches += 1

            pbar.set_postfix({
                'loss':  f"{loss_dict['total']:.4f}",
                'mse':   f"{loss_dict['latent_mse']:.4f}",
                'l1':    f"{loss_dict['spectral_l1']:.4f}",
                'cmpct': f"{loss_dict.get('compact', 0):.3f}",
            })

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        return epoch_losses

    # ------------------------------------------------------------------
    # Validate epoch
    # ------------------------------------------------------------------

    @torch.no_grad()
    def validate_epoch(self, epoch):
        """Validate and compute bona-fide vs. attack reconstruction gap."""
        self.model.eval()
        epoch_losses = {
            'total': 0, 'latent_mse': 0, 'spectral_l1': 0, 'center_loss': 0
        }
        num_batches = 0

        bf_l1_all  = []
        atk_l1_all = []

        pbar = tqdm(self.val_loader,
                    desc=f"Epoch {epoch+1}/{self.num_epochs} [Val]")

        for dwt_concat, hh_band, labels, metadata in pbar:
            dwt_concat = dwt_concat.to(self.device, non_blocking=True)
            hh_band    = hh_band.to(self.device, non_blocking=True)
            labels_dev = labels.to(self.device, non_blocking=True)

            with autocast('cuda', enabled=self.use_amp):
                B = dwt_concat.size(0)
                noise_level = torch.full((B,), cfg.NOISE_LEVEL, device=self.device)
                output = self.model(dwt_concat, hh_band, noise_level)

                # Compute loss on bona-fide only (for early stopping signal)
                bona_fide_mask = (labels_dev == 0)
                if bona_fide_mask.any():
                    bf_output = {
                        'z0':          output['z0'][bona_fide_mask],
                        'latent_mse':  output['latent_mse'][bona_fide_mask],
                        'spectral_l1': output['spectral_l1'][bona_fide_mask],
                    }
                    _, loss_dict = self.criterion(bf_output, labels=None)
                else:
                    loss_dict = {
                        'total': 0, 'latent_mse': 0,
                        'spectral_l1': 0, 'center_loss': 0
                    }

            # Track reconstruction gap (key health metric!)
            sl1 = output['spectral_l1'].cpu()
            lbl = labels.cpu()
            bf_l1_all.extend(sl1[lbl == 0].tolist())
            atk_l1_all.extend(sl1[lbl > 0].tolist())

            for key in epoch_losses:
                epoch_losses[key] += loss_dict.get(key, 0)
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss_dict['total']:.6f}"})

        for key in epoch_losses:
            epoch_losses[key] /= max(num_batches, 1)

        # Reconstruction gap: positive means attacks have higher error (good!)
        if bf_l1_all and atk_l1_all:
            recon_gap = float(np.mean(atk_l1_all) - np.mean(bf_l1_all))
            bf_std    = float(np.std(bf_l1_all)) + 1e-8
            sep_ratio = recon_gap / bf_std  # separation in units of std
        else:
            recon_gap  = 0.0
            sep_ratio  = 0.0

        epoch_losses['recon_gap']  = recon_gap
        epoch_losses['sep_ratio']  = sep_ratio
        return epoch_losses

    # ------------------------------------------------------------------
    # Full training loop
    # ------------------------------------------------------------------

    def train(self):
        """Full training loop with two-phase backbone strategy."""
        print(f"\n{'='*60}")
        print("TRAINING START  (Two-Phase Strategy)")
        print(f"{'='*60}")
        trainable = sum(p.numel() for p in self.model.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.model.parameters())
        print(f"  Total parameters:     {total:,}")
        print(f"  Trainable parameters: {trainable:,}  ({100*trainable/total:.1f}%)")
        print(f"  Backbone frozen:      Phase-1 (epochs 1-{PHASE2_EPOCH})")
        print(f"  Phase-2 unfreeze:     layer3+layer4 at epoch {PHASE2_EPOCH+1}")
        print(f"  Base LR:              {self.base_lr:.1e}")
        print(f"  Phase-2 LR:           {self.base_lr/10:.1e}")
        print(f"  DELTA (compact):      {self.criterion.delta}")
        print(f"  Device: {self.device}")
        print(f"  Mixed Precision: {self.use_amp}")
        print(f"  Train batches: {len(self.train_loader)}")
        if self.val_loader:
            print(f"  Val batches: {len(self.val_loader)}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(self.num_epochs):

            # ---- Phase-2 transition ----
            if epoch == PHASE2_EPOCH and not self._phase2_activated:
                self._activate_phase2()

            train_losses = self.train_epoch(epoch)

            # ---- Centroid initialization after epoch 0 ----
            # Run whenever compact loss (DELTA) or repulsion (EPSILON) is active.
            # Both losses depend on the fixed bona-fide centroid.
            if epoch == 0 and not self.criterion.center_initialized:
                needs_centroid = (getattr(cfg, 'DELTA', 0) > 0 or
                                  getattr(cfg, 'EPSILON', 0) > 0)
                if needs_centroid:
                    self._initialize_centroid()
                else:
                    print("  [Centroid Init] Skipped (DELTA=0 and EPSILON=0).")

            self.history['train_loss'].append(train_losses['total'])
            self.history['train_latent_mse'].append(train_losses['latent_mse'])
            self.history['train_spectral_l1'].append(train_losses['spectral_l1'])
            self.history['train_center_loss'].append(train_losses['center_loss'])
            self.history['train_compact'].append(train_losses.get('compact', 0))
            self.history['train_repulsion'].append(train_losses.get('repulsion', 0))
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

            val_losses = {
                'total': 0, 'latent_mse': 0, 'spectral_l1': 0,
                'center_loss': 0, 'recon_gap': 0.0, 'sep_ratio': 0.0
            }
            if self.val_loader:
                val_losses = self.validate_epoch(epoch)

            self.history['val_loss'].append(val_losses['total'])
            self.history['val_latent_mse'].append(val_losses['latent_mse'])
            self.history['val_spectral_l1'].append(val_losses['spectral_l1'])
            self.history['val_center_loss'].append(val_losses['center_loss'])
            self.history['val_recon_gap'].append(val_losses.get('recon_gap', 0.0))

            self.scheduler.step()

            elapsed = time.time() - start_time
            phase_tag = "[Phase-2]" if self._phase2_activated else "[Phase-1]"
            print(f"\n  Epoch {epoch+1}/{self.num_epochs} {phase_tag} "
                  f"({elapsed/60:.1f} min elapsed)")
            print(f"    Train Loss: {train_losses['total']:.6f} "
                  f"(MSE: {train_losses['latent_mse']:.6f}, "
                  f"L1: {train_losses['spectral_l1']:.6f}, "
                  f"Compact: {train_losses.get('compact', 0):.4f})")
            if self.val_loader:
                gap      = val_losses.get('recon_gap', 0.0)
                sep      = val_losses.get('sep_ratio', 0.0)
                gap_sign = '+' if gap >= 0 else ''
                health   = '[GOOD ✓]' if gap > 0 else '[BAD  ✗ - no separation]'
                print(f"    Val Loss:   {val_losses['total']:.6f} "
                      f"(ReconGap: {gap_sign}{gap*1000:.2f}e-3, "
                      f"SepRatio: {sep:.2f}std  {health})")
            print(f"    LR: {self.optimizer.param_groups[0]['lr']:.2e}")

            # Save best model based on val bona-fide loss
            current_loss = val_losses['total'] if self.val_loader else train_losses['total']
            if current_loss < self.best_val_loss:
                self.best_val_loss = current_loss
                self.patience_counter = 0
                self.save_checkpoint('best_model.pth', epoch, current_loss)
                print(f"    [✓] New best model saved (loss: {current_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"    No improvement ({self.patience_counter}/{cfg.EARLY_STOP_PATIENCE})")

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth',
                                     epoch, current_loss)

            if self.patience_counter >= cfg.EARLY_STOP_PATIENCE:
                print(f"\n  Early stopping at epoch {epoch+1}")
                break

            print()

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"  Total time: {total_time/60:.1f} minutes")
        print(f"  Best loss: {self.best_val_loss:.6f}")
        print(f"{'='*60}")

        self.save_history()
        return self.history

    def save_checkpoint(self, filename, epoch, loss):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'scaler_state_dict':    self.scaler.state_dict(),
            'loss':                 loss,
            'best_val_loss':        self.best_val_loss,
            'criterion_state_dict': self.criterion.state_dict(),  # saves centroid buffer
        }, path)

    def load_checkpoint(self, filename='best_model.pth'):
        path = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        if 'criterion_state_dict' in checkpoint:
            self.criterion.load_state_dict(checkpoint['criterion_state_dict'])
            print(f"  Centroid loaded (initialized={self.criterion.center_initialized})")
        print(f"  Loaded checkpoint from epoch {checkpoint['epoch']+1} "
              f"(loss: {checkpoint['loss']:.6f})")
        return checkpoint['epoch']

    def save_history(self):
        path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"  Training history saved to {path}")
