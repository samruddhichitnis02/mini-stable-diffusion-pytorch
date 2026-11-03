# DATA 266 Assignment 9: Quantitative Metrics & Summary

- **Student ID (SID4)**: `2122`
- **Random Seed**: `2122`
- **Execution Device**: `cuda`
- **Pretrained VAE Model**: `stabilityai/sd-vae-ft-mse`
- **VAE Latent Scaling Factor ($s$)**: `0.18215`
- **VAE Test Reconstruction MSE**: `0.000886`

## Latent Diffusion Architecture & Training

- **U-Net Parameter Count**: `9,159,300`
- **Latent Input Dimension**: `[4, 16, 16]`
- **Base Channels**: `64` (Multipliers: `[1, 2, 4]`)
- **Diffusion Timesteps ($T$)**: `1000`
- **Beta Schedule**: Linear `[0.0001, 0.02]`
- **Training Epochs**: `50`
- **Learning Rate**: `2e-05`
- **Classifier-Free Guidance Training Dropout ($p_{uncond}$)**: `0.1`
- **Null Label Index**: `10` (10 classes + 1 null)
- **Initial Training Loss (Epoch 1)**: `0.415875`
- **Final Training Loss (Epoch 50)**: `0.097723`
- **Total Training Duration**: `4984.66` seconds (83.08 mins)

## Sampling Evaluation

- **CFG Guidance Scale ($w$)**: `3.0`
- **Grid Sample Count**: `64` (Fashion-MNIST classes 0–9 repeated)
