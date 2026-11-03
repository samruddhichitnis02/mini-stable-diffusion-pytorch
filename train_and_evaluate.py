import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import sys
import csv
import json
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from scheduler import LatentDiffusionScheduler
from vae_pipeline import (
    get_vae,
    verify_vae_reconstruction,
    visualize_forward_diffusion_latents,
    encode_to_latents
)
from dataset import (
    get_raw_fashion_mnist_loaders,
    prepare_and_cache_latents,
    get_latent_dataloaders
)
from model import LatentUNet
from sampler import generate_64_sample_grid, compare_guidance_scales

SEED = 2122
SID4 = 2122
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class TeeLogger:
    """Tees standard output to both console and a log file."""
    def __init__(self, filename="RUN_LOG.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

def set_seed(seed=2122):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_ldm(model, scheduler, train_loader, epochs=50, lr=2e-5, p_uncond=0.1, null_label=10, 
              checkpoint_path="ldm_fashion_mnist.pt", loss_csv_path="training_loss.csv"):
    """
    Trains the Latent Diffusion Model conditional U-Net to predict noise
    using Classifier-Free Guidance label dropout.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_history = []
    
    print(f"\n==========================================")
    print(f"Starting LDM Training: {epochs} Epochs, LR: {lr}, Device: {DEVICE}")
    print(f"CFG Label Dropout Probability: {p_uncond}, Null Label: {null_label}")
    print(f"==========================================")
    
    total_start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        ep_start = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        for latents, labels in train_loader:
            latents = latents.to(DEVICE)
            labels = labels.to(DEVICE)
            B = latents.size(0)
            
            # Sample random timesteps t in [0, T-1]
            t = torch.randint(0, scheduler.num_timesteps, (B,), device=DEVICE).long()
            
            # Sample random Gaussian noise
            noise = torch.randn_like(latents)
            
            # Forward diffusion noise injection
            z_t, _ = scheduler.q_sample(latents, t, noise=noise)
            
            # Classifier-Free Guidance: randomly drop labels to null_label with prob p_uncond
            drop_mask = torch.rand(B, device=DEVICE) < p_uncond
            train_labels = torch.where(drop_mask, torch.tensor(null_label, device=DEVICE), labels)
            
            # Model prediction
            noise_pred = model(z_t, t, train_labels)
            
            # MSE loss on noise
            loss = F.mse_loss(noise_pred, noise)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
        avg_loss = total_loss / num_batches
        loss_history.append((epoch, avg_loss))
        ep_time = time.time() - ep_start
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {avg_loss:.6f} - Time: {ep_time:.2f}s")
            
    total_time = time.time() - total_start_time
    print(f"\nTraining completed in {total_time:.2f} seconds ({total_time/60:.2f} minutes).")
    
    # Save checkpoint
    torch.save({
        'epoch': epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss_history': loss_history,
        'hyperparameters': {
            'epochs': epochs,
            'lr': lr,
            'p_uncond': p_uncond,
            'null_label': null_label,
            'base_channels': 64,
            'channel_multipliers': (1, 2, 4),
            'total_params': sum(p.numel() for p in model.parameters())
        }
    }, checkpoint_path)
    print(f"Saved trained model checkpoint to '{checkpoint_path}'.")
    
    # Save CSV
    with open(loss_csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Training_MSE_Loss"])
        for ep, l_val in loss_history:
            writer.writerow([ep, f"{l_val:.6f}"])
    print(f"Saved training loss history to '{loss_csv_path}'.")
    
    # Plot Loss Curve
    plot_loss_curve(loss_history, output_path="training_loss.png")
    return loss_history, total_time

def plot_loss_curve(loss_history, output_path="training_loss.png"):
    epochs = [x[0] for x in loss_history]
    losses = [x[1] for x in loss_history]
    
    plt.figure(figsize=(9, 5))
    plt.plot(epochs, losses, marker='o', markersize=3, color='#1f77b4', linewidth=2, label="Training MSE Loss")
    plt.title("Conditional Latent U-Net Training Loss (50 Epochs, LR: 2e-5)", fontsize=13, fontweight='bold', pad=10)
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("MSE Loss $(\\epsilon - \\hat{\\epsilon})^2$", fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved loss curve to '{output_path}'.")

def save_metrics_summary(metrics, json_path="metrics.json", md_path="metrics.md"):
    with open(json_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved quantitative metrics to '{json_path}'.")

    with open(md_path, 'w') as f:
        f.write("# DATA 266 Assignment 9: Quantitative Metrics & Summary\n\n")
        f.write(f"- **Student ID (SID4)**: `{metrics['student_id']}`\n")
        f.write(f"- **Random Seed**: `{metrics['seed']}`\n")
        f.write(f"- **Execution Device**: `{metrics['device']}`\n")
        f.write(f"- **Pretrained VAE Model**: `{metrics['vae_model']}`\n")
        f.write(f"- **VAE Latent Scaling Factor ($s$)**: `{metrics['latent_scaling_factor']}`\n")
        f.write(f"- **VAE Test Reconstruction MSE**: `{metrics['vae_reconstruction_mse']:.6f}`\n\n")
        f.write("## Latent Diffusion Architecture & Training\n\n")
        f.write(f"- **U-Net Parameter Count**: `{metrics['unet_parameters']:,}`\n")
        f.write(f"- **Latent Input Dimension**: `{metrics['latent_shape']}`\n")
        f.write(f"- **Base Channels**: `{metrics['base_channels']}` (Multipliers: `{metrics['channel_multipliers']}`)\n")
        f.write(f"- **Diffusion Timesteps ($T$)**: `{metrics['diffusion_timesteps']}`\n")
        f.write(f"- **Beta Schedule**: Linear `[{metrics['beta_start']}, {metrics['beta_end']}]`\n")
        f.write(f"- **Training Epochs**: `{metrics['epochs']}`\n")
        f.write(f"- **Learning Rate**: `{metrics['learning_rate']}`\n")
        f.write(f"- **Classifier-Free Guidance Training Dropout ($p_{{uncond}}$)**: `{metrics['p_uncond']}`\n")
        f.write(f"- **Null Label Index**: `{metrics['null_label']}` (10 classes + 1 null)\n")
        f.write(f"- **Initial Training Loss (Epoch 1)**: `{metrics['initial_loss']:.6f}`\n")
        f.write(f"- **Final Training Loss (Epoch 50)**: `{metrics['final_loss']:.6f}`\n")
        f.write(f"- **Total Training Duration**: `{metrics['training_duration_seconds']:.2f}` seconds ({metrics['training_duration_minutes']:.2f} mins)\n\n")
        f.write("## Sampling Evaluation\n\n")
        f.write(f"- **CFG Guidance Scale ($w$)**: `{metrics['guidance_scale']}`\n")
        f.write(f"- **Grid Sample Count**: `{metrics['sample_grid_count']}` (Fashion-MNIST classes 0–9 repeated)\n")
    print(f"Saved markdown metrics report to '{md_path}'.")

def main():
    sys.stdout = TeeLogger("RUN_LOG.txt")

    print(f"==================================================")
    print(f"DATA 266 Assignment 9: Latent Diffusion Model (LDM)")
    print(f"Student ID (SID4): {SID4} | Seed: {SEED} | Device: {DEVICE}")
    print(f"PyTorch Version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"==================================================")
    
    set_seed(SEED)
    
    # 1. Load Pretrained VAE
    vae = get_vae(device=DEVICE)
    raw_train_loader, raw_test_loader = get_raw_fashion_mnist_loaders(root="./data", batch_size=64)
    
    # 2. VAE Reconstruction & Verification
    print("\n--- Part 1: VAE Reconstruction Verification ---")
    mse_score = verify_vae_reconstruction(vae, raw_test_loader, num_images=8, output_path="vae_reconstruction.png")
    
    # 3. Forward Diffusion in Latent Space
    print("\n--- Part 2: Forward Diffusion in Latent Space ---")
    scheduler = LatentDiffusionScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device=DEVICE)
    visualize_forward_diffusion_latents(vae, scheduler, raw_test_loader, 
                                        timesteps=[0, 200, 500, 800, 1000],
                                        output_path="forward_diffusion_latents_decoded.png")
    
    # 4. Prepare Latents Dataset
    print("\n--- Part 3: Dataset Preparation & Caching ---")
    train_latents_ds, test_latents_ds = prepare_and_cache_latents(vae, root="./data", cache_dir=".", batch_size=64, device=DEVICE)
    train_loader, test_loader = get_latent_dataloaders(train_latents_ds, test_latents_ds, batch_size=128)
    
    # 5. Conditional U-Net Initialization
    print("\n--- Part 4: Conditional U-Net Initialization & Training ---")
    model = LatentUNet(in_channels=4, out_channels=4, base_channels=64, num_classes=11).to(DEVICE)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Instantiated LatentUNet: {total_params:,} parameters.")
    
    # Train model for 50 epochs
    loss_history, total_train_time = train_ldm(
        model=model,
        scheduler=scheduler,
        train_loader=train_loader,
        epochs=50,
        lr=2e-5,
        p_uncond=0.1,
        null_label=10,
        checkpoint_path="ldm_fashion_mnist.pt",
        loss_csv_path="training_loss.csv"
    )
    
    # 6. Reverse Sampling & Evaluation
    print("\n--- Part 5: Sampling with Classifier-Free Guidance ---")
    generate_64_sample_grid(
        model=model,
        vae=vae,
        scheduler=scheduler,
        guidance_scale=3.0,
        output_path="sample_grid_64.png",
        seed=SEED
    )
    
    print("\n--- Part 6: CFG Guidance Scale Ablation ---")
    compare_guidance_scales(
        model=model,
        vae=vae,
        scheduler=scheduler,
        output_path="guidance_scale_comparison.png",
        seed=SEED
    )

    # 7. Save Metrics Summary
    metrics = {
        'student_id': SID4,
        'seed': SEED,
        'device': DEVICE,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU',
        'vae_model': 'stabilityai/sd-vae-ft-mse',
        'latent_scaling_factor': 0.18215,
        'vae_reconstruction_mse': mse_score,
        'latent_shape': [4, 16, 16],
        'unet_parameters': total_params,
        'base_channels': 64,
        'channel_multipliers': [1, 2, 4],
        'diffusion_timesteps': 1000,
        'beta_start': 1e-4,
        'beta_end': 0.02,
        'epochs': 50,
        'learning_rate': 2e-5,
        'p_uncond': 0.1,
        'null_label': 10,
        'initial_loss': loss_history[0][1],
        'final_loss': loss_history[-1][1],
        'training_duration_seconds': total_train_time,
        'training_duration_minutes': total_train_time / 60.0,
        'guidance_scale': 3.0,
        'sample_grid_count': 64
    }
    save_metrics_summary(metrics, json_path="metrics.json", md_path="metrics.md")
    
    print("\n==================================================")
    print("All Assignment 9 experiments completed successfully!")
    print(f"==================================================")

if __name__ == "__main__":
    main()
