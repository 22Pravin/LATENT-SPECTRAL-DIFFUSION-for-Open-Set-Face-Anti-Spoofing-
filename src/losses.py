"""
Loss functions for the DWT Denoising Autoencoder.

Combined loss:
  L_total = alpha  * L_latent
          + beta   * L_spectral
          + delta  * L_compact      (bona-fide only)
          + epsilon * L_repulsion   (known attacks only, when TRAIN_WITH_ATTACKS=True)

Where:
  L_latent    = MSE(z0, z_recon)             - latent denoising reconstruction error
  L_spectral  = L1(HH_recon, HH_orig)        - HH band reconstruction error
  L_compact   = mean ||z0_bf - centroid||^2  - Deep SVDD: pulls bona-fide to tight cluster
  L_repulsion = mean max(0, M - ||z0_atk - centroid||^2)
                                             - pushes known attacks beyond margin M

OPEN-SET ONE-CLASS PROTOCOL:
  - Bona-fide only for compact loss (unsupervised)
  - Known attack (print1) ONLY for repulsion margin — unknown attacks still unseen
  - Centroid is a FIXED buffer (initialized after epoch 0, never updated by gradient)
  - This prevents the classic collapse where centroid chases the encoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CombinedPADLoss(nn.Module):
    """
    Combined training loss for the DWT Autoencoder.

    L_total = alpha   * MSE(z0, z_recon)      [latent denoising]
            + beta    * L1(HH_recon, HH_orig)  [spectral HH reconstruction]
            + delta   * Compact(z0_bf)          [Deep SVDD: bona-fide cluster]
            + epsilon * Repulsion(z0_atk)       [push known attacks beyond margin]

    Compact loss (bona-fide only, Deep SVDD style):
      Applied ONLY to samples with label == 0.
      Pulls bona-fide embeddings toward a FIXED centroid.

    Repulsion loss (known attacks only, when TRAIN_WITH_ATTACKS=True):
      Applied ONLY to samples with label > 0.
      Pushes known attack embeddings BEYOND repulsion_margin from centroid.
      Formula: mean max(0, margin - ||z0_atk - centroid||^2)
      (0 when attacks are already far enough away)

    Open-set guarantee: unknown attacks (print2, cce, hp) are never in
    the training DataLoader, so they are never seen by this loss.
    """

    def __init__(self, alpha=1.0, beta=1.0, gamma=0.0, delta=0.1,
                 epsilon=0.1, repulsion_margin=10.0, latent_dim=512):
        """
        Args:
            alpha:             Weight for latent MSE loss.
            beta:              Weight for spectral L1 loss.
            gamma:             Unused (kept for backward compat), always 0.
            delta:             Weight for compact centroid loss (bona-fide only).
            epsilon:           Weight for repulsion loss (known attacks only).
            repulsion_margin:  Squared-L2 margin M; attacks already beyond M
                               contribute 0 repulsion gradient.
            latent_dim:        Dimension of the latent vector z0.
        """
        super().__init__()
        self.alpha             = alpha
        self.beta              = beta
        self.gamma             = 0.0   # Always disabled
        self.delta             = delta
        self.epsilon           = epsilon
        self.repulsion_margin  = repulsion_margin

        # Fixed bona-fide centroid (Deep SVDD style).
        # This is a buffer (not a parameter), so it is NEVER updated by
        # backprop. Set once from data after epoch 0, then frozen.
        self.register_buffer('center', torch.zeros(latent_dim))
        self.center_initialized = False

    def initialize_center(self, mean_z0: torch.Tensor):
        """
        Set the centroid from the mean bona-fide z0 of epoch 0.
        Called once by the Trainer after epoch 0 completes.

        Args:
            mean_z0: (latent_dim,) tensor — mean of all bona-fide z0
                     embeddings collected during epoch 0.
        """
        with torch.no_grad():
            # Deep SVDD safeguard: avoid near-zero centroid components
            eps = 0.01
            mean_z0 = mean_z0.to(self.center.device)
            mean_z0[(mean_z0 <  eps) & (mean_z0 >= 0)] =  eps
            mean_z0[(mean_z0 > -eps) & (mean_z0 <  0)] = -eps
            self.center.copy_(mean_z0)
        self.center_initialized = True
        print(f"  [CombinedPADLoss] Centroid initialized "
              f"(norm={self.center.norm().item():.4f}). Now frozen.")

    def forward(self, model_output, labels=None):
        """
        Compute the combined loss.

        Args:
            model_output: Dict from DWTAutoencoder.forward():
                - z0:          (B, latent_dim)
                - latent_mse:  (B,) per-sample latent MSE
                - spectral_l1: (B,) per-sample spectral L1
            labels: (B,) int tensor of class labels.
                    0 = bona-fide, >0 = attack.
                    When None (val bona-fide pass), all samples treated as bona-fide.

        Returns:
            total_loss: scalar
            loss_dict:  dict with individual component values for logging
        """
        z0            = model_output['z0']
        latent_mse    = model_output['latent_mse']
        spectral_l1   = model_output['spectral_l1']
        device        = z0.device

        # --- Build bona-fide and attack masks ---
        if labels is not None:
            bf_mask  = (labels == 0)
            atk_mask = (labels  > 0)
        else:
            # Validation pass: all samples assumed bona-fide
            bf_mask  = torch.ones(len(z0), dtype=torch.bool, device=device)
            atk_mask = torch.zeros(len(z0), dtype=torch.bool, device=device)

        # ---- 1. Reconstruction losses (all samples) ----
        # Compute on bona-fide samples only — attacks may have high
        # reconstruction error intentionally; including them in the
        # reconstruction loss would push the model to reconstruct attacks well.
        if bf_mask.any():
            latent_loss   = latent_mse[bf_mask].mean()
            spectral_loss = spectral_l1[bf_mask].mean()
        else:
            latent_loss   = latent_mse.mean()
            spectral_loss = spectral_l1.mean()

        # ---- 2. Compact centroid loss (bona-fide only) ----
        # Pulls bona-fide z0 toward the FIXED centroid (Deep SVDD).
        # Active only after centroid is initialized (epoch >= 1).
        if self.center_initialized and bf_mask.any():
            z0_bf        = z0[bf_mask]
            dist_sq_bf   = torch.sum((z0_bf - self.center.detach()) ** 2, dim=1)
            compact_loss = dist_sq_bf.mean()
        else:
            compact_loss = torch.tensor(0.0, device=device)

        # ---- 3. Repulsion loss (known attacks only) ----
        # Pushes attack z0 BEYOND repulsion_margin from the centroid.
        # Uses hinge: max(0, margin - dist^2) so no gradient once far enough.
        # Only active after centroid is initialized.
        if self.center_initialized and self.epsilon > 0 and atk_mask.any():
            z0_atk       = z0[atk_mask]
            dist_sq_atk  = torch.sum((z0_atk - self.center.detach()) ** 2, dim=1)
            repulsion_loss = torch.clamp(
                self.repulsion_margin - dist_sq_atk, min=0.0
            ).mean()
        else:
            repulsion_loss = torch.tensor(0.0, device=device)

        # ---- Total loss ----
        total_loss = (self.alpha   * latent_loss
                    + self.beta    * spectral_loss
                    + self.delta   * compact_loss
                    + self.epsilon * repulsion_loss)

        loss_dict = {
            'total':       total_loss.item(),
            'latent_mse':  latent_loss.item(),
            'spectral_l1': spectral_loss.item(),
            'center_loss': compact_loss.item(),
            'compact':     compact_loss.item(),
            'repulsion':   repulsion_loss.item(),
        }

        return total_loss, loss_dict


class LatentMSELoss(nn.Module):
    """Standalone latent MSE loss."""

    def forward(self, z0, z_recon):
        return F.mse_loss(z_recon, z0)


class SpectralL1Loss(nn.Module):
    """Standalone spectral L1 loss on HH band."""

    def forward(self, hh_recon, hh_original):
        return F.l1_loss(hh_recon, hh_original)
