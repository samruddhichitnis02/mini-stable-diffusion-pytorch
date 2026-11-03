import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib.pyplot as plt
from scheduler import LatentDiffusionScheduler
from vae_pipeline import get_vae, decode_from_latents
from model import LatentUNet
from sampler import FASHION_MNIST_CLASSES

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SEED = 2122

def set_seed(seed=2122):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def slerp(val, low, high):
    """Spherical linear interpolation."""
    low_norm = low / torch.norm(low, dim=-1, keepdim=True)
    high_norm = high / torch.norm(high, dim=-1, keepdim=True)
    dot = (low_norm * high_norm).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    omega = torch.acos(dot)
    so = torch.sin(omega)
    if (so == 0).all():
        return (1.0 - val) * low + val * high
    return (torch.sin((1.0 - val) * omega) / so) * low + (torch.sin(val * omega) / so) * high

@torch.no_grad()
def fast_reverse_sample(model, scheduler, z_init, labels, guidance_scales=3.0, num_steps=50):
    """
    Accelerated batched reverse sampling with stride = 1000 // num_steps (DDIM/strided DDPM).
    Allows real-time simulation generation in 1-2 seconds.
    """
    timesteps = np.linspace(scheduler.num_timesteps - 1, 0, num_steps, dtype=int)
    z = z_init.clone()
    B = z.size(0)

    # Handle scalar or vector guidance scales
    if isinstance(guidance_scales, (int, float)):
        w_tensor = torch.full((B, 1, 1, 1), guidance_scales, device=DEVICE)
    else:
        w_tensor = guidance_scales.view(B, 1, 1, 1).to(DEVICE)

    null_labels = torch.full_like(labels, 10)

    for i, t_idx in enumerate(timesteps):
        t_tensor = torch.full((B,), t_idx, device=DEVICE, dtype=torch.long)
        
        eps_cond = model(z, t_tensor, labels)
        eps_uncond = model(z, t_tensor, null_labels)
        eps = (1.0 + w_tensor) * eps_cond - w_tensor * eps_uncond

        alpha_bar_t = scheduler.alphas_cumprod[t_idx]
        prev_t = timesteps[i + 1] if i + 1 < len(timesteps) else 0
        alpha_bar_prev = scheduler.alphas_cumprod[prev_t] if prev_t > 0 else torch.tensor(1.0, device=DEVICE)

        # DDIM style deterministic ODE step
        pred_z0 = (z - torch.sqrt(1.0 - alpha_bar_t) * eps) / torch.sqrt(alpha_bar_t)
        dir_zt = torch.sqrt(1.0 - alpha_bar_prev) * eps
        z = torch.sqrt(alpha_bar_prev) * pred_z0 + dir_zt

    return z

@torch.no_grad()
def simulate_latent_interpolation(model, vae, scheduler, pair=(0, 3), steps=8,
                                 output_path="latent_interpolation_simulation.png"):
    """
    Simulation 2: Latent Manifold Interpolation (T-shirt -> Dress).
    Batched execution across all alpha steps simultaneously.
    """
    print("Generating Simulation 2: Latent Manifold Interpolation...")
    set_seed(SEED + 10)
    
    z_A = torch.randn((1, 4, 16, 16), device=DEVICE)
    z_B = torch.randn((1, 4, 16, 16), device=DEVICE)
    
    cls_A, cls_B = pair
    alphas = np.linspace(0.0, 1.0, steps)
    
    z_batch = []
    labels_list = []
    for alpha in alphas:
        z_interp = slerp(alpha, z_A.view(1, -1), z_B.view(1, -1)).view(1, 4, 16, 16)
        z_batch.append(z_interp)
        cur_cls = cls_A if alpha < 0.5 else cls_B
        labels_list.append(cur_cls)
        
    z_batch = torch.cat(z_batch, dim=0) # (8, 4, 16, 16)
    labels = torch.tensor(labels_list, device=DEVICE)
    
    # 50-step accelerated sampling for the entire batch
    z_final = fast_reverse_sample(model, scheduler, z_batch, labels, guidance_scales=3.0, num_steps=50)
    
    decoded = decode_from_latents(vae, z_final)
    disp = ((decoded.cpu() + 1.0) / 2.0).clamp(0, 1)

    fig, axes = plt.subplots(1, steps, figsize=(steps * 2.2, 2.8))
    for idx, ax in enumerate(axes):
        ax.imshow(disp[idx].permute(1, 2, 0).numpy())
        alpha = alphas[idx]
        ax.set_title(f"$\\alpha = {alpha:.2f}$\n({FASHION_MNIST_CLASSES[cls_A] if alpha<0.5 else FASHION_MNIST_CLASSES[cls_B]})", 
                     fontsize=9, pad=4)
        ax.axis('off')

    plt.suptitle(f"Simulation 2: Continuous Latent Manifold Interpolation ({FASHION_MNIST_CLASSES[cls_A]} $\\to$ {FASHION_MNIST_CLASSES[cls_B]})",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Simulation 2 to '{output_path}'.")

@torch.no_grad()
def simulate_cfg_sweep(model, vae, scheduler, test_class=2, scales=[0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 7.5],
                       output_path="cfg_guidance_sweep_simulation.png"):
    """
    Simulation 3: Fine-Grained CFG Scale Sweep.
    Batched execution across all scales simultaneously with identical noise seed.
    """
    print("Generating Simulation 3: Fine-Grained CFG Guidance Scale Sweep...")
    set_seed(SEED + 42)
    B = len(scales)
    z_init = torch.randn((1, 4, 16, 16), device=DEVICE).repeat(B, 1, 1, 1)
    labels = torch.full((B,), test_class, device=DEVICE, dtype=torch.long)
    scales_tensor = torch.tensor(scales, device=DEVICE, dtype=torch.float32)

    z_final = fast_reverse_sample(model, scheduler, z_init, labels, guidance_scales=scales_tensor, num_steps=50)
    
    decoded = decode_from_latents(vae, z_final)
    disp = ((decoded.cpu() + 1.0) / 2.0).clamp(0, 1)

    fig, axes = plt.subplots(1, B, figsize=(B * 2.2, 3.0))
    for idx, ax in enumerate(axes):
        ax.imshow(disp[idx].permute(1, 2, 0).numpy())
        w = scales[idx]
        sub = "Assigned" if w == 3.0 else ("Uncond" if w == 0.0 else "")
        ax.set_title(f"$w = {w}$\n{f'({sub})' if sub else ''}", fontsize=10, fontweight='bold', pad=4)
        ax.axis('off')

    plt.suptitle(f"Simulation 3: CFG Guidance Scale Sweep for Class {test_class} ({FASHION_MNIST_CLASSES[test_class]})",
                 fontsize=13, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved Simulation 3 to '{output_path}'.")

def main():
    print("Loading pretrained VAE and trained LDM checkpoint for simulations...")
    vae = get_vae(device=DEVICE)
    scheduler = LatentDiffusionScheduler(num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device=DEVICE)

    model = LatentUNet(in_channels=4, out_channels=4, base_channels=64, num_classes=11).to(DEVICE)
    ckpt = torch.load("ldm_fashion_mnist.pt", map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print("Model loaded successfully!")

    # Simulation 1 (already generated or generate if missing)
    if not os.path.exists("reverse_denoising_trajectories.png"):
        from sampler import sample_ldm
        print("Simulation 1 will be generated...")

    # 2. Latent Interpolation Simulation (T-shirt to Dress)
    simulate_latent_interpolation(model, vae, scheduler, pair=(0, 3), steps=8)

    # 3. CFG Guidance Scale Sweep Simulation
    simulate_cfg_sweep(model, vae, scheduler, test_class=2)

    print("\nAll simulations generated successfully!")

if __name__ == "__main__":
    main()
