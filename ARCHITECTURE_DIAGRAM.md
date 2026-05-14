# DWTAutoencoder Architecture Dataflow

The following diagram maps out the flow of data through the entire DWT Autoencoder Presentation Attack Detection model, from the raw image input down to the calculated loss components.

```mermaid
%%{init: {'theme': 'base', 'themeVariables': { 'primaryColor': '#1e1e2f', 'primaryTextColor': '#fff', 'primaryBorderColor': '#7C3AED', 'lineColor': '#8B5CF6', 'secondaryColor': '#1F2937', 'tertiaryColor': '#374151'}}}%%

graph TD
    %% Define Styles
    classDef input fill:#059669,stroke:#047857,stroke-width:2px,color:#fff
    classDef preproc fill:#2563EB,stroke:#1D4ED8,stroke-width:2px,color:#fff
    classDef nnModule fill:#7C3AED,stroke:#6D28D9,stroke-width:2px,color:#fff
    classDef tensor fill:#4B5563,stroke:#374151,stroke-width:1px,color:#E5E7EB
    classDef loss fill:#DC2626,stroke:#B91C1C,stroke-width:2px,color:#fff
    
    %% Inputs & Preprocessing (dataset.py)
    subgraph Data Loading & Preprocessing
        A[Input Face Image <br> 128x128x3 RGB]:::input --> B[2D-DWT Transform <br> 'haar' wavelet]:::preproc
        B -->|Extract Bands| C1(LL Band <br> 64x64x3)
        B -->|Extract Bands| C2(LH Band <br> 64x64x3)
        B -->|Extract Bands| C3(HL Band <br> 64x64x3)
        B -->|Extract Bands| C4(HH Band <br> 64x64x3)
        
        C1 & C2 & C3 & C4 -->|Concatenate| D[DWT Concat Tensor <br> 12x64x64]:::tensor
        C4 -->|Ground Truth Target| E[Original HH Tensor <br> 3x64x64]:::tensor
    end

    %% Model Architecture (model.py)
    subgraph DWTAutoencoder Model
        D --> Enc[ResNetDWTEncoder <br> Modded ResNet-18]:::nnModule
        Enc --> Z0[Clean Latent Vector z0 <br> 512-dim]:::tensor
        
        %% Diffusion Logic
        Z0 --> Diff[Forward Diffusion <br> sqrt1-s^2 * z0 + s * noise]:::preproc
        Noise[Random Noise Level s <br> Scalar 0.1 to 0.8]:::input --> Diff
        Noise --> NEmb[NoiseEmbedding <br> Sinusoidal]:::nnModule
        NEmb --> NEmbT[Noise Embed Vector <br> 64-dim]:::tensor
        
        Diff --> Znoisy[Noisy Latent zT <br> 512-dim]:::tensor
        
        Znoisy --> Den[Denoiser MLP <br> 2x 1024-dim layers]:::nnModule
        NEmbT --> Den
        
        Den --> Zrecon[Reconstructed Latent z0 <br> 512-dim]:::tensor
        
        Zrecon --> Dec[SpectralDecoder <br> Transposed CNN]:::nnModule
        Dec --> HHrecon[Reconstructed HH Band <br> 3x64x64]:::tensor
    end

    %% Loss Calculation (losses.py)
    subgraph Loss Computation
        Zrecon -.-> LMSE((Latent MSE <br> L2 Distance)):::loss
        Z0 -.-> LMSE
        
        HHrecon -.-> SL1((Spectral L1 <br> L1 Distance)):::loss
        E -.-> SL1
        
        Z0 -.-> CR((Compact-Repulsion Loss <br> Deep SVDD)):::loss
    end

    %% Inference Output
    subgraph Inference Anomaly Score
        LMSE -->|Weight: λ| Final[Final Anomaly Score <br> λ*LMSE + 1-λ*SL1]:::preproc
        SL1 -->|Weight: 1-λ| Final
    end
```
