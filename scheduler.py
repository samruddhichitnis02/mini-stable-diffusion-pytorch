import torch
import torch.nn as nn
import numpy as np

class LatentDiffusionScheduler:
    """
    Gaussian Diffusion Scheduler operating in the latent space.
    Implements a linear beta schedule with T = 1000 diffusion timesteps,
    beta_t in [10^-4, 0.02].
    """
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device='cuda'):
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([torch.tensor([1.0], device=self.device), self.alphas_cumprod[:-1]])

        # Calculations for diffusion q(z_t | z_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(z_{t-1} | z_t, z_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    def to(self, device):
        self.device = torch.device(device)
        self.betas = self.betas.to(self.device)
        self.alphas = self.alphas.to(self.device)
        self.alphas_cumprod = self.alphas_cumprod.to(self.device)
        self.alphas_cumprod_prev = self.alphas_cumprod_prev.to(self.device)
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(self.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(self.device)
        self.posterior_variance = self.posterior_variance.to(self.device)
        self.sqrt_recip_alphas = self.sqrt_recip_alphas.to(self.device)
        return self

    def q_sample(self, z_0, t, noise=None):
        """
        Closed-form forward diffusion process:
        q(z_t | z_0) = N(z_t; sqrt(alpha_bar_t) * z_0, (1 - alpha_bar_t) * I)
        """
        if noise is None:
            noise = torch.randn_like(z_0)

        # Handle tensor indexing
        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)

        z_t = sqrt_alpha_bar * z_0 + sqrt_one_minus_alpha_bar * noise
        return z_t, noise

    def p_sample_step(self, model, z_t, t_idx, y, guidance_scale=3.0, null_label=10):
        """
        Reverse denoising step with Classifier-Free Guidance (CFG).
        Combines conditional and unconditional noise predictions:
        eps = (1 + w) * eps_cond - w * eps_uncond
        """
        B = z_t.size(0)
        t_tensor = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        # Conditional prediction
        eps_cond = model(z_t, t_tensor, y)

        if guidance_scale > 0:
            # Unconditional prediction using null label
            y_uncond = torch.full_like(y, null_label)
            eps_uncond = model(z_t, t_tensor, y_uncond)

            # CFG combination: eps = (1 + w) * eps_cond - w * eps_uncond
            eps = (1.0 + guidance_scale) * eps_cond - guidance_scale * eps_uncond
        else:
            eps = eps_cond

        # DDPM mean computation:
        # mu = (1 / sqrt(alpha_t)) * (z_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps)
        beta_t = self.betas[t_idx]
        sqrt_recip_alpha_t = self.sqrt_recip_alphas[t_idx]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t_idx]

        model_mean = sqrt_recip_alpha_t * (z_t - (beta_t / sqrt_one_minus_alpha_bar_t) * eps)

        if t_idx == 0:
            return model_mean
        else:
            posterior_var = self.posterior_variance[t_idx]
            noise = torch.randn_like(z_t)
            return model_mean + torch.sqrt(posterior_var) * noise

