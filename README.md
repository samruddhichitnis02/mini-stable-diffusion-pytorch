# Latent Diffusion Model (LDM) on Fashion-MNIST with Classifier-Free Guidance

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8%2Bcu128-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20(RTX%203060)-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Dataset](https://img.shields.io/badge/Dataset-FashionMNIST-yellow.svg)](https://github.com/zalandoresearch/fashion-mnist)
[![VAE](https://img.shields.io/badge/VAE-stabilityai%2Fsd--vae--ft--mse-purple.svg)](https://huggingface.co/stabilityai/sd-vae-ft-mse)
[![Notebook](https://img.shields.io/badge/Jupyter-Executed-orange.svg)](Latent_Diffusion_FashionMNIST.ipynb)

> A modular PyTorch implementation of a **Latent Diffusion Model (LDM)** operating in a compressed **Variational Autoencoder (VAE)** latent space with **Self-Attention conditioning** and **Classifier-Free Guidance (CFG)** on Fashion-MNIST.

---

## Live Generative Simulations

### 1. Reverse Diffusion Denoising Simulation ($t = 1000 \to 0$)
Watch 4 clothing categories (*Trouser, Pullover, Dress, Sneaker*) synthesize simultaneously from pure Gaussian noise in latent space into high-fidelity garments with Classifier-Free Guidance ($w = 3$):

![Reverse Diffusion Denoising Simulation](reverse_diffusion_simulation.gif)

---

### 2. Continuous Latent Space Metamorphosis (Morphing Simulation)
Seamless spherical linear interpolation (Slerp) along the learned latent manifold looping between:
**T-shirt/top $\to$ Dress $\to$ Pullover $\to$ Sneaker $\to$ T-shirt/top**:

![Continuous Latent Morphing Simulation](latent_morph_simulation.gif)

---

## Personal Parameters & Step 0 Alignment

| Parameter | Value | Description |
| :--- | :---: | :--- |
| **Student ID (`SID4`)** | `2122` | Last 4 digits of SJSU student ID |
| **Random Seed (`SEED`)** | `2122` | Applied deterministically across Python, NumPy, PyTorch & CUDA |
| **Pretrained VAE** | `stabilityai/sd-vae-ft-mse` | AutoencoderKL ($f=8$ downsampling, 4 latent channels) |
| **Latent Scaling Factor** | $s = 0.18215$ | Latent normalization: $z_0 = 0.18215 \cdot \mathcal{E}(x)$ |
| **U-Net Capacity** | `9,159,300` | Base 64 ch, multipliers (1, 2, 4), self-attention at $16\times 16$ & bottleneck |
| **CFG Guidance Scale** | $w = 3.0$ | Score extrapolation: $\epsilon = (1 + w)\epsilon_{\text{cond}} - w\epsilon_{\text{uncond}}$ |
| **Training Budget** | 50 Epochs | Adam optimizer, LR = $2\times 10^{-5}$, $p_{\text{uncond}} = 0.1$ |

---

## Visual Results & Assignment Deliverables

### Part 1: VAE Latent Compression & Reconstruction
Fashion-MNIST ($128\times 128\times 3$) is compressed $64\times$ into $4\times 16\times 16$ latents. The VAE achieves near-lossless reconstruction:
- **Test Reconstruction MSE:** **`0.000886`**

![VAE Reconstruction](vae_reconstruction.png)

---

### Part 2: Forward Diffusion in Latent Space
Progressive Gaussian noise injection into scaled latents $z_0$ across timesteps $t \in \{0, 200, 500, 800, 1000\}$, decoded back into image space via $\hat{x} = \mathcal{D}(z_t / s)$:

![Forward Latent Diffusion Decoded](forward_diffusion_latents_decoded.png)

---

### Part 4: 50-Epoch Training Loss Convergence
The Conditional Latent U-Net was trained for 50 epochs to predict noise $\epsilon$. The loss dropped by **76.5%** monotonically:
- **Initial Loss (Epoch 1):** `0.415875`
- **Midpoint Loss (Epoch 25):** `0.106896`
- **Final Loss (Epoch 50):** **`0.097723`** (MSE)

![Training Loss Curve](training_loss.png)

---

### Part 5: 64-Image Conditional Sample Grid ($w = 3$)
Complete $8 \times 8$ grid of 64 generated clothing samples repeating Fashion-MNIST classes 0 through 9 using Classifier-Free Guidance ($w = 3$):

![64 Sample Grid](sample_grid_64.png)

---

### Part 6: Classifier-Free Guidance Scale Ablation ($w \in \{0.0, 1.0, 3.0\}$)
Comparing the impact of guidance scale $w$ with identical latent noise seeds across all 10 Fashion-MNIST classes:

![Guidance Scale Comparison](guidance_scale_comparison.png)

- **$w = 0.0$ (Unconditional):** High diversity, but softer outlines and slight class ambiguity.
- **$w = 1.0$ (Moderate):** Enhanced contrast and clear garment definition.
- **$w = 3.0$ (Assigned Optimum):** Sharpest contours, clean silhouettes, and zero background artifacts.

---

## Detailed Generative Simulations

### Simulation 1: Step-by-Step Reverse Trajectory Filmstrip
Intermediate snapshots during reverse diffusion ($t = 1000 \to 0$) across 4 garment classes:

![Reverse Denoising Trajectories](reverse_denoising_trajectories.png)

### Simulation 2: Continuous Semantic Manifold Interpolation
Continuous morphing between T-shirt and Dress across 8 steps:

![Latent Interpolation Simulation](latent_interpolation_simulation.png)

### Simulation 3: Fine-Grained CFG Sweep ($w \in [0.0, 7.5]$)
Guidance trade-off curve on Pullover:

![CFG Sweep Simulation](cfg_guidance_sweep_simulation.png)

---

## Project Structure

```text
├── reverse_diffusion_simulation.gif      # Live animated reverse denoising GIF
├── latent_morph_simulation.gif           # Live animated class morphing GIF
├── Latent_Diffusion_FashionMNIST.ipynb   # Executed Jupyter Notebook with outputs
├── scheduler.py                         # Latent diffusion scheduler (T=1000, linear beta)
├── vae_pipeline.py                      # Pretrained VAE loader, scaling (s=0.18215) & decoder
├── model.py                             # Conditional Latent U-Net with self-attention (9.16M params)
├── dataset.py                           # GPU-accelerated latent precomputation & DataLoader
├── sampler.py                           # Reverse sampling with CFG (w=3) & grid generator
├── train_and_evaluate.py                # End-to-end training, evaluation & metrics logging
├── generate_simulations.py              # Generative simulations & trajectory visualizers
├── create_animated_gifs.py              # Script compiling animated GIFs
├── training_loss.csv                    # 50-epoch numerical loss history
├── training_log.txt                     # Formatted training log table
├── RUN_LOG.txt                          # Complete terminal execution log
├── metrics.json / metrics.md            # Quantitative evaluation summaries
├── *.png                                # High-resolution result figures & plots
└── Question.pdf                         # Original homework specification
```

---

## How to Run & Reproduce

```bash
# 1. Activate environment
conda activate my_env

# 2. Run complete training, evaluation and sampling pipeline
python train_and_evaluate.py

# 3. Generate simulation figures and animated GIFs
python generate_simulations.py
python create_animated_gifs.py

# 4. Compile executed Jupyter Notebook
python build_notebook.py
```
