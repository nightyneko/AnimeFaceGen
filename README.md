# AnimeFaceGen — Self-Attention DCGAN (PyTorch, 64×64)

A compact PyTorch implementation of a Self-Attention DCGAN that generates 64×64 anime faces.  
It trains on the **Kaggle `splcher/animefacedataset`** (via `kagglehub`) and includes **FID** and **Inception Score** evaluation, animated training previews, and exported computational graphs for G & D.

---

## Features

- **Architecture:** DCGAN backbone + **Self-Attention** blocks in **G** and **D**
- **Image Size:** 64×64 (RGB) with `tanh` output (`[-1, 1]`)
- **Training:** `BCEWithLogitsLoss`, Adam (β₁=0.5)
- **Metrics:** FID + IS using `torch-fidelity`
- **Utilities:** Sample grids, training animation, TorchViz graphs

---

## Dataset

- **Source:** Kaggle → `splcher/animefacedataset`
- **Access:** Downloaded automatically via `kagglehub` (requires Kaggle API creds if private/rate-limited)

> Real images are resized to **64×64** before FID/IS to match the generator output.

---

## Model Overview

### Generator (G)
- Input: noise **z** ∈ ℝ^(300×1×1)
- Blocks (typical DCGAN upsampling):
  - ConvTranspose2d → 4×4 (ch 512 = 64×8)
  - ConvTranspose2d → 8×8 (ch 256 = 64×4)
  - **SelfAttention(256)**
  - ConvTranspose2d → 16×16 (ch 128 = 64×2)
  - ConvTranspose2d → 32×32 (ch 64)
  - ConvTranspose2d → 64×64 (ch 3), **tanh**
- Activations: **PReLU** + **BatchNorm2d**
- Init: DCGAN-style normal init (mean=0.0, std=0.02)

### Discriminator (D)
- Input: 3×64×64
- Blocks:
  - Conv → 32×32 (ch 64), **PReLU**
  - **SelfAttention(64)**
  - Conv → 16×16 (ch 128), **BatchNorm2d**, PReLU
  - Conv → 8×8 (ch 256), **BatchNorm2d**, PReLU
  - Conv → 4×4 (ch 512), **BatchNorm2d**, PReLU
  - Conv → 1×1 (ch 1) → logits
- Loss: `BCEWithLogitsLoss`

---
