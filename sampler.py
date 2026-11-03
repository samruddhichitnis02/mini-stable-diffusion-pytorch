import os
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from tqdm import tqdm
from vae_pipeline import decode_from_latents

FASHION_MNIST_CLASSES = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

@torch.no_grad()
def sample_ldm(model, vae, scheduler, labels, guidance_scale=3.0, null_label=10, 
               latent_shape=(4, 16, 16), seed=2122, desc="Sampling LDM"):
    """
    Reverse latent diffusion sampling with Classifier-Free Guidance.
    Formula: eps = (1 + w) * eps_cond - w * eps_uncond, with w = guidance_scale.
    """
    device = scheduler.device
    model.eval()
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    B = labels.size(0)
    # Start from pure Gaussian noise in latent space
    z_t = torch.randn((B, *latent_shape), device=device)

    for t_idx in tqdm(reversed(range(scheduler.num_timesteps)), total=scheduler.num_timesteps, desc=desc):
        z_t = scheduler.p_sample_step(
            model=model,
            z_t=z_t,
            t_idx=t_idx,
            y=labels,
            guidance_scale=guidance_scale,
            null_label=null_label
        )

    # Decode final latents z_0 to pixel images
    decoded_images = decode_from_latents(vae, z_t)
    return decoded_images, z_t

def generate_64_sample_grid(model, vae, scheduler, guidance_scale=3.0, 
                            output_path="sample_grid_64.png", seed=2122):
    """
    Section 5: Generate a 64-image conditional sample grid using
    Fashion-MNIST class labels 0-9 repeated, guidance scale w = 3.
    """
    device = scheduler.device
    num_samples = 64
    labels = torch.tensor([i % 10 for i in range(num_samples)], device=device)

    print(f"Generating 64-image sample grid with CFG guidance scale w = {guidance_scale}...")
    images, final_latents = sample_ldm(
        model=model,
        vae=vae,
        scheduler=scheduler,
        labels=labels,
        guidance_scale=guidance_scale,
        seed=seed,
        desc=f"Reverse Sampling (w={guidance_scale})"
    )

    # Convert [-1, 1] to [0, 1]
    images_01 = ((images.cpu() + 1.0) / 2.0).clamp(0, 1)

    # Plot 8x8 grid with class names
    fig, axes = plt.subplots(8, 8, figsize=(16, 18))
    for i in range(64):
        row = i // 8
        col = i % 8
        ax = axes[row, col]
        img_np = images_01[i].permute(1, 2, 0).numpy()
        ax.imshow(img_np)
        class_idx = labels[i].item()
        ax.set_title(f"{class_idx}: {FASHION_MNIST_CLASSES[class_idx]}", fontsize=8, pad=3)
        ax.axis('off')

    plt.suptitle(
        f"Latent Diffusion Model: 64-Image Conditional Sample Grid\n"
        f"Classifier-Free Guidance ($w = {guidance_scale}$, $\\epsilon = (1+w)\\epsilon_{{cond}} - w\\epsilon_{{uncond}}$)",
        fontsize=15, fontweight='bold', y=0.99
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved 64-image sample grid to '{output_path}'.")
    return images_01

def compare_guidance_scales(model, vae, scheduler, output_path="guidance_scale_comparison.png", seed=2122):
    """
    Compares unconditional (w=0), mild (w=1.0), and assigned (w=3.0) CFG guidance scales
    across all 10 FashionMNIST classes with matched noise seeds.
    """
    device = scheduler.device
    scales = [0.0, 1.0, 3.0]
    num_classes = 10
    labels = torch.arange(num_classes, device=device)

    results = {}
    for w in scales:
        print(f"Generating samples for CFG scale w = {w}...")
        imgs, _ = sample_ldm(
            model=model,
            vae=vae,
            scheduler=scheduler,
            labels=labels,
            guidance_scale=w,
            seed=seed,
            desc=f"CFG w={w}"
        )
        results[w] = ((imgs.cpu() + 1.0) / 2.0).clamp(0, 1)

    fig, axes = plt.subplots(len(scales), num_classes, figsize=(num_classes * 1.8, len(scales) * 2.2))
    for row_idx, w in enumerate(scales):
        imgs = results[w]
        for col_idx in range(num_classes):
            ax = axes[row_idx, col_idx]
            ax.imshow(imgs[col_idx].permute(1, 2, 0).numpy())
            ax.axis('off')
            if row_idx == 0:
                ax.set_title(f"{col_idx}: {FASHION_MNIST_CLASSES[col_idx]}", fontsize=9, pad=4)
        axes[row_idx, 0].text(-0.35, 0.5, f"w = {w}\n({'(Assigned)' if w==3.0 else 'Ablation'})",
                              transform=axes[row_idx, 0].transAxes, fontsize=11, fontweight='bold',
                              va='center', ha='right')

    plt.suptitle("Classifier-Free Guidance Comparison Across Scale $w \\in \\{0.0, 1.0, 3.0\\}$",
                 fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved guidance scale comparison to '{output_path}'.")

