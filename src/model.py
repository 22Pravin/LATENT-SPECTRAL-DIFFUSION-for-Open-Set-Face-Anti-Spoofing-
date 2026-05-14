"""
Model architecture for Autoencoder-based Open-Set PAD.

Components:
1. LightweightEncoder  — CNN that encodes concatenated DWT bands → latent z₀
2. NoiseEmbedding      — Sinusoidal embedding of the noise level
3. Denoiser            — MLP that denoises z_T* → ẑ₀ (reverse process)
4. SpectralDecoder     — Transposed CNN that decodes ẑ₀ → reconstructed HH band
5. DWTAutoencoder      — Full model combining all components
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models


class ResNetDWTEncoder(nn.Module):
    """
    ResNet-18 based encoder adapted to process 12-channel DWT sub-bands.
    Extracts deep semantic features into a compact latent vector z₀.

    Input:  (B, 12, 64, 64) — 4 DWT bands × 3 RGB channels
    Output: (B, latent_dim) — latent vector z₀
    """

    def __init__(self, in_channels=12, latent_dim=512):
        super().__init__()

        # Load pre-trained ResNet-18 (ImageNet weights)
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Modify the first conv layer to accept 12 channels instead of 3
        original_conv1 = resnet.conv1
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv1.out_channels,
            kernel_size=original_conv1.kernel_size,
            stride=original_conv1.stride,
            padding=original_conv1.padding,
            bias=original_conv1.bias is not None
        )

        # Initialize the new 12-channel weights by duplicating the pre-trained 3-channel weights 4 times
        with torch.no_grad():
            self.conv1.weight.data = original_conv1.weight.data.repeat(1, in_channels // 3, 1, 1) / (in_channels // 3)
            if self.conv1.bias is not None:
                self.conv1.bias.data = original_conv1.bias.data

        # Copy the remaining layers from the pre-trained ResNet
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool

        # Replace the final fully connected layer to output our target latent_dim
        self.fc = nn.Sequential(
            nn.Linear(resnet.fc.in_features, latent_dim),
            nn.LayerNorm(latent_dim),
        )

    def freeze_backbone(self):
        """
        Freeze all ResNet layers except the final FC projection.
        This prevents the powerful ImageNet features from overfitting to attack faces.
        Only the compact FC head is trained, forcing it to learn a tight bona-fide manifold.
        """
        for module in [self.conv1, self.bn1, self.layer1, self.layer2,
                       self.layer3, self.layer4, self.avgpool]:
            for param in module.parameters():
                param.requires_grad = False
        print("  [Encoder] Backbone frozen. Only FC head is trainable.")

    def unfreeze_top_layers(self):
        """
        Phase-2 partial unfreeze: unfreeze only layer3, layer4, avgpool, and fc.
        Keeps conv1, bn1, layer1, layer2 frozen to preserve low-level ImageNet
        features. This is gentler than a full unfreeze and reduces overfitting
        risk when fine-tuning with a small bona-fide dataset.

        Use this at epoch ~20 with a 10x smaller LR than the head.
        """
        # Keep early layers frozen
        for module in [self.conv1, self.bn1, self.layer1, self.layer2]:
            for param in module.parameters():
                param.requires_grad = False
        # Unfreeze high-level layers
        for module in [self.layer3, self.layer4, self.avgpool, self.fc]:
            for param in module.parameters():
                param.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  [Encoder] Phase-2: layer3+layer4+fc unfrozen "
              f"({n_trainable:,} trainable encoder params).")

    def unfreeze_backbone(self):
        """Unfreeze ALL parameters (use only for full fine-tuning)."""
        for param in self.parameters():
            param.requires_grad = True
        print("  [Encoder] Backbone fully unfrozen.")

    def forward(self, x):
        """
        Args:
            x: (B, 12, 64, 64) concatenated DWT bands
        Returns:
            z0: (B, latent_dim) latent vector
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        z0 = self.fc(x)
        return z0


class NoiseEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for the noise level.
    Maps a scalar noise level to a vector representation,
    allowing the denoiser to be noise-level-aware.
    """

    def __init__(self, embed_dim=64):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, noise_level):
        """
        Args:
            noise_level: (B,) tensor of noise levels in [0, 1]
        Returns:
            embedding: (B, embed_dim)
        """
        device = noise_level.device
        half_dim = self.embed_dim // 2
        emb_scale = math.log(10000) / (half_dim - 1)
        emb = torch.exp(
            torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_scale
        )
        emb = noise_level.unsqueeze(1) * emb.unsqueeze(0)  # (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # (B, embed_dim)
        return emb


class Denoiser(nn.Module):
    """
    MLP-based denoiser (reverse process).
    Takes noisy latent z_T* + noise level embedding and
    reconstructs the clean latent ẑ₀.

    Input:  z_noisy (B, latent_dim), noise_level (B,)
    Output: z_reconstructed (B, latent_dim)
    """

    def __init__(self, latent_dim=512, noise_embed_dim=64, hidden_dim=1024):
        super().__init__()

        self.noise_embed = NoiseEmbedding(noise_embed_dim)

        self.net = nn.Sequential(
            nn.Linear(latent_dim + noise_embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # Residual projection
        self.residual_proj = nn.Linear(latent_dim + noise_embed_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, latent_dim)

    def forward(self, z_noisy, noise_level):
        """
        Args:
            z_noisy: (B, latent_dim) noisy latent vector
            noise_level: (B,) noise level scalars
        Returns:
            z_recon: (B, latent_dim) reconstructed latent vector
        """
        # Embed noise level
        noise_emb = self.noise_embed(noise_level)  # (B, noise_embed_dim)

        # Concatenate noisy latent with noise embedding
        x = torch.cat([z_noisy, noise_emb], dim=1)  # (B, latent_dim + noise_embed_dim)

        # Forward with residual connection
        h = self.net(x)
        h = h + self.residual_proj(x)  # Residual

        z_recon = self.output(h)
        return z_recon


class SpectralDecoder(nn.Module):
    """
    Transposed CNN decoder that reconstructs the HH (high-frequency)
    wavelet band from the latent vector.

    This is critical for detecting presentation attacks, as print and
    screen attacks introduce characteristic artifacts in the HH band
    (moiré patterns, printing dots, pixel grid patterns).

    Input:  ẑ₀ (B, latent_dim)
    Output: HH_reconstructed (B, 3, 64, 64)
    """

    def __init__(self, latent_dim=512, out_channels=3):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4),
            nn.LayerNorm(256 * 4 * 4),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            # (256, 4, 4) → (128, 8, 8)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # (128, 8, 8) → (64, 16, 16)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),

            # (64, 16, 16) → (32, 32, 32)
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),

            # (32, 32, 32) → (3, 64, 64)
            nn.ConvTranspose2d(32, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        """
        Args:
            z: (B, latent_dim) latent vector
        Returns:
            hh_recon: (B, 3, 64, 64) reconstructed HH band
        """
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4)
        hh_recon = self.decoder(x)
        return hh_recon


class DWTAutoencoder(nn.Module):
    """
    Complete DWT-based Denoising Autoencoder for Open-Set PAD.

    Pipeline:
        1. Encode concatenated DWT bands → latent z₀
        2. Forward diffusion: z₀ → z_T* (add noise)
        3. Reverse/Denoise: z_T* → ẑ₀
        4. Compute latent MSE: ||z₀ - ẑ₀||²
        5. Decode ẑ₀ → HH_reconstructed
        6. Compute spectral L₁: ||HH_recon - HH_original||₁

    Training: Only on bona fide (real) face images.
    Inference: High reconstruction error → presentation attack.
    """

    def __init__(self, latent_dim=512, noise_embed_dim=64,
                 encoder_channels=[32, 64, 128, 256],
                 denoiser_hidden=1024):
        super().__init__()

        self.latent_dim = latent_dim

        # Components
        self.encoder = ResNetDWTEncoder(
            in_channels=12,  # 4 DWT bands × 3 channels
            latent_dim=latent_dim,
        )

        self.denoiser = Denoiser(
            latent_dim=latent_dim,
            noise_embed_dim=noise_embed_dim,
            hidden_dim=denoiser_hidden,
        )

        self.spectral_decoder = SpectralDecoder(
            latent_dim=latent_dim,
            out_channels=3,
        )

    def forward_diffusion(self, z0, noise_level):
        """
        Forward diffusion process: add Gaussian noise to latent vector.

        Args:
            z0: (B, latent_dim) clean latent vector
            noise_level: (B,) noise level in [0, 1]

        Returns:
            z_noisy: (B, latent_dim) noisy latent
            noise: (B, latent_dim) the noise that was added
        """
        noise = torch.randn_like(z0)

        # Expand noise_level for broadcasting: (B,) → (B, 1)
        nl = noise_level.unsqueeze(1)

        # z_noisy = sqrt(1 - σ²) · z₀ + σ · noise
        z_noisy = torch.sqrt(1.0 - nl ** 2) * z0 + nl * noise

        return z_noisy, noise

    def forward(self, dwt_bands_concat, hh_original, noise_level=None):
        """
        Full forward pass through the autoencoder.

        Args:
            dwt_bands_concat: (B, 12, 64, 64) concatenated DWT bands
            hh_original: (B, 3, 64, 64) original HH band (for loss)
            noise_level: (B,) noise levels. If None, sampled randomly.

        Returns:
            dict with:
                - z0: original latent
                - z_noisy: noisy latent
                - z_recon: reconstructed latent
                - hh_recon: reconstructed HH band
                - latent_mse: per-sample latent MSE scores
                - spectral_l1: per-sample spectral L1 scores
        """
        B = dwt_bands_concat.size(0)
        device = dwt_bands_concat.device

        # 1. Encode → latent z₀
        z0 = self.encoder(dwt_bands_concat)

        # 2. Sample noise level if not provided
        if noise_level is None:
            noise_level = torch.FloatTensor(B).uniform_(0.1, 0.8).to(device)

        # 3. Forward diffusion: z₀ → z_T*
        z_noisy, noise = self.forward_diffusion(z0, noise_level)

        # 4. Reverse / Denoise: z_T* → ẑ₀
        z_recon = self.denoiser(z_noisy, noise_level)

        # 5. Spectral decode: ẑ₀ → HH_reconstructed
        hh_recon = self.spectral_decoder(z_recon)

        # 6. Compute scores (per-sample)
        latent_mse = F.mse_loss(z_recon, z0, reduction='none').mean(dim=1)  # (B,)
        spectral_l1 = F.l1_loss(hh_recon, hh_original, reduction='none').mean(dim=(1, 2, 3))  # (B,)

        return {
            'z0': z0,
            'z_noisy': z_noisy,
            'z_recon': z_recon,
            'hh_recon': hh_recon,
            'latent_mse': latent_mse,
            'spectral_l1': spectral_l1,
        }

    def compute_anomaly_score(self, dwt_bands_concat, hh_original,
                               noise_level_val=0.5, lambda_fusion=0.5):
        """
        Compute the final anomaly score for inference.

        Args:
            dwt_bands_concat: (B, 12, 64, 64) concatenated DWT bands
            hh_original: (B, 3, 64, 64) original HH band
            noise_level_val: fixed noise level for evaluation
            lambda_fusion: fusion weight λ

        Returns:
            s_final: (B,) final anomaly scores
            s_latent: (B,) latent MSE scores
            s_spectral: (B,) spectral L1 scores
        """
        B = dwt_bands_concat.size(0)
        device = dwt_bands_concat.device

        noise_level = torch.full((B,), noise_level_val, device=device)

        with torch.no_grad():
            output = self.forward(dwt_bands_concat, hh_original, noise_level)

        s_latent = output['latent_mse']
        s_spectral = output['spectral_l1']

        # Weighted fusion: S_final = λ · S_latent + (1-λ) · S_spectral
        s_final = lambda_fusion * s_latent + (1 - lambda_fusion) * s_spectral

        return s_final, s_latent, s_spectral

    def count_parameters(self):
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config=None):
    """
    Factory function to build the model with default or custom config.

    Returns:
        model: DWTAutoencoder instance
    """
    if config is None:
        from . import config as cfg
        model = DWTAutoencoder(
            latent_dim=cfg.LATENT_DIM,
            noise_embed_dim=cfg.NOISE_EMBED_DIM,
            encoder_channels=cfg.ENCODER_CHANNELS,
            denoiser_hidden=cfg.DENOISER_HIDDEN,
        )
    else:
        model = DWTAutoencoder(
            latent_dim=config.LATENT_DIM,
            noise_embed_dim=config.NOISE_EMBED_DIM,
            encoder_channels=config.ENCODER_CHANNELS,
            denoiser_hidden=config.DENOISER_HIDDEN,
        )

    return model
