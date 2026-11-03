import os
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
from diffusers import AutoencoderKL

LATENT_SCALE_FACTOR = 0.18215

def get_vae(device='cuda', model_name='stabilityai/sd-vae-ft-mse'):
    """
    Loads and freezes the pretrained VAE encoder-decoder model.
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Loading pretrained VAE '{model_name}' on {device}...")
    vae = AutoencoderKL.from_pretrained(model_name).to(device)
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae

def get_fashion_mnist_transform():
    """
    FashionMNIST images are 1x28x28 grayscale.
    SD-VAE expects 3 channels and downsamples 8x.
    Resizing to 128x128 produces exact 16x16 latents with 4 channels.
    Pixel values are normalized to [-1, 1].
    """
    return T.Compose([
        T.Resize((128, 128), interpolation=T.InterpolationMode.BILINEAR),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

@torch.no_grad()
def encode_to_latents(vae, images, sample=True):
    """
    Encodes images into scaled latent representation:
    z = s * encoder(x), where s = 0.18215
    """
    dist = vae.encode(images).latent_dist
    latents = dist.sample() if sample else dist.mode()
    return latents * LATENT_SCALE_FACTOR

@torch.no_grad()
def decode_from_latents(vae, latents):
    """
    Decodes scaled latents back to image space:
    x_rec = decoder(z / s), where s = 0.18215
    Clamped to [-1, 1].
    """
    unscaled_latents = latents / LATENT_SCALE_FACTOR
    decoded = vae.decode(unscaled_latents).sample
    return torch.clamp(decoded, -1.0, 1.0)

@torch.no_grad()
def verify_vae_reconstruction(vae, dataloader, num_images=8, output_path="vae_reconstruction.png"):
    """
    Verifies reconstruction capability by passing scaled latents through the decoder
    and measuring MSE relative to the original inputs.
    Saves a comparison grid of original vs reconstructed images.
    """
    device = next(vae.parameters()).device
    images, labels = next(iter(dataloader))
    images = images[:num_images].to(device)

    # 1. Encode with scaling factor
    latents = encode_to_latents(vae, images, sample=False)

    # 2. Decode back through unscaling
    reconstructions = decode_from_latents(vae, latents)

    # 3. Measure MSE
    mse = torch.mean((reconstructions - images) ** 2).item()
    print(f"[VAE Verification] Reconstruction MSE on {num_images} sample images: {mse:.6f}")

    # Plot Original vs Reconstructed
    fig, axes = plt.subplots(2, num_images, figsize=(num_images * 2, 4.5))
    orig_disp = ((images.cpu() + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).numpy()
    rec_disp = ((reconstructions.cpu() + 1.0) / 2.0).clamp(0, 1).permute(0, 2, 3, 1).numpy()

    for i in range(num_images):
        axes[0, i].imshow(orig_disp[i])
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title("Original", fontsize=11, fontweight='bold', loc='left')

        axes[1, i].imshow(rec_disp[i])
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title(f"Reconstructed\nMSE: {mse:.5f}", fontsize=11, fontweight='bold', loc='left')

    plt.suptitle(f"VAE Latent Compression & Reconstruction (Scale Factor s = {LATENT_SCALE_FACTOR})",
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved VAE reconstruction comparison to '{output_path}'.")
    return mse

@torch.no_grad()
def visualize_forward_diffusion_latents(vae, scheduler, dataloader,
                                       timesteps=[0, 200, 500, 800, 1000],
                                       output_path="forward_diffusion_latents_decoded.png"):
    """
    Section 2: Implement noise injection function and visualize decoded image
    representations from noisy latents z_t across timesteps t in {0, 200, 500, 800, 1000}.
    """
    device = next(vae.parameters()).device
    images, _ = next(iter(dataloader))
    img = images[0:1].to(device)

    # Clean latent z_0
    z_0 = encode_to_latents(vae, img, sample=False)

    torch.manual_seed(2122)
    noise = torch.randn_like(z_0)

    decoded_images = []
    titles = []

    for t_val in timesteps:
        if t_val == 0:
            z_t = z_0
            title = "t = 0\n(Clean Latent)"
        elif t_val >= scheduler.num_timesteps:
            # Full Gaussian noise limit
            t_tensor = torch.tensor([scheduler.num_timesteps - 1], device=device)
            z_t, _ = scheduler.q_sample(z_0, t_tensor, noise=noise)
            title = f"t = {t_val}\n(Pure Noise)"
        else:
            t_tensor = torch.tensor([t_val], device=device)
            z_t, _ = scheduler.q_sample(z_0, t_tensor, noise=noise)
            title = f"t = {t_val}\n(Noisy Latent)"

        # Decode noisy latent through VAE decoder
        decoded_img = decode_from_latents(vae, z_t)
        disp = ((decoded_img.squeeze().cpu() + 1.0) / 2.0).clamp(0, 1).permute(1, 2, 0).numpy()
        decoded_images.append(disp)
        titles.append(title)

    fig, axes = plt.subplots(1, len(timesteps), figsize=(len(timesteps) * 2.8, 3.5))
    for idx, ax in enumerate(axes):
        ax.imshow(decoded_images[idx])
        ax.set_title(titles[idx], fontsize=11, fontweight='bold', pad=6)
        ax.axis('off')

    plt.suptitle("Forward Diffusion in VAE Latent Space: Decoded Images $x = \\mathcal{D}(z_t / s)$",
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved forward diffusion latent visualization to '{output_path}'.")

