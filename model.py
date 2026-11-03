import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPositionEmbeddings(nn.Module):
    """
    Standard sinusoidal positional embedding for diffusion timesteps.
    PE(t, 2i)   = sin(t / (10000^(2i / dim)))
    PE(t, 2i+1) = cos(t / (10000^(2i / dim)))
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None].float() * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class AttentionBlock(nn.Module):
    """
    Multi-head self-attention block with GroupNorm and residual connection.
    Used at 16x16 resolution and at the bottleneck (4x4).
    """
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads

        num_groups = min(32, channels)
        while channels % num_groups != 0:
            num_groups //= 2
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        norm_x = self.norm(x)
        qkv = self.qkv(norm_x)
        q, k, v = qkv.chunk(3, dim=1)

        # Reshape to (B, num_heads, H*W, head_dim)
        q = q.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        k = k.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)
        v = v.view(B, self.num_heads, self.head_dim, H * W).permute(0, 1, 3, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).contiguous().view(B, C, H, W)
        return x + self.proj_out(out)

class ResnetBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU, Conv2d, and condition projection.
    """
    def __init__(self, in_channels, out_channels, time_emb_dim):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        num_groups_in = min(32, in_channels)
        while in_channels % num_groups_in != 0:
            num_groups_in //= 2
        self.norm1 = nn.GroupNorm(num_groups_in, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.cond_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels)
        )

        num_groups_out = min(32, out_channels)
        while out_channels % num_groups_out != 0:
            num_groups_out //= 2
        self.norm2 = nn.GroupNorm(num_groups_out, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x, cond):
        h = self.conv1(self.act1(self.norm1(x)))
        cond_bias = self.cond_proj(cond)[:, :, None, None]
        h = h + cond_bias
        h = self.conv2(self.act2(self.norm2(h)))
        return h + self.shortcut(x)

class LatentUNet(nn.Module):
    """
    Conditional U-Net Denoiser operating on VAE latent space.
    Specifications:
    - Input: Latent space of the VAE (4 channels, 16x16)
    - Base channels: 64, channel multipliers: (1, 2, 4) -> (64, 128, 256)
    - Sinusoidal time embedding + MLP
    - Label embeddings: 10 Fashion-MNIST classes + 1 null label (total 11)
    - Residual blocks + GroupNorm + SiLU
    - Self-attention at 16x16 + bottleneck (4x4)
    - Downsampling, upsampling, and skip connections
    """
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, num_classes=11):
        super().__init__()
        self.base_channels = base_channels
        self.time_emb_dim = base_channels * 4  # 256
        self.num_classes = num_classes

        # Timestep embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(base_channels),
            nn.Linear(base_channels, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Label embedding (10 classes + 1 null class at index 10)
        self.label_emb = nn.Embedding(num_classes, self.time_emb_dim)

        # Combined conditioning projection
        self.cond_proj = nn.Sequential(
            nn.Linear(self.time_emb_dim * 2, self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(self.time_emb_dim, self.time_emb_dim)
        )

        # Initial convolution
        c1 = base_channels      # 64
        c2 = base_channels * 2  # 128
        c3 = base_channels * 4  # 256

        self.conv_in = nn.Conv2d(in_channels, c1, kernel_size=3, padding=1)

        # Downsampling stages
        # Level 0 (16x16, 64 channels) with self-attention
        self.down0_res1 = ResnetBlock(c1, c1, self.time_emb_dim)
        self.down0_attn = AttentionBlock(c1)
        self.down0_down = nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1) # -> 8x8

        # Level 1 (8x8, 128 channels)
        self.down1_res1 = ResnetBlock(c1, c2, self.time_emb_dim)
        self.down1_res2 = ResnetBlock(c2, c2, self.time_emb_dim)
        self.down1_down = nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1) # -> 4x4

        # Level 2 (4x4, 256 channels)
        self.down2_res1 = ResnetBlock(c2, c3, self.time_emb_dim)
        self.down2_res2 = ResnetBlock(c3, c3, self.time_emb_dim)

        # Bottleneck (4x4, 256 channels) with self-attention
        self.mid_res1 = ResnetBlock(c3, c3, self.time_emb_dim)
        self.mid_attn = AttentionBlock(c3)
        self.mid_res2 = ResnetBlock(c3, c3, self.time_emb_dim)

        # Upsampling stages
        # Level 2 up (4x4 -> 8x8)
        self.up2_res = ResnetBlock(c3 + c3, c3, self.time_emb_dim)
        self.up2_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c3, c2, kernel_size=3, padding=1)
        ) # -> 8x8, 128 channels

        # Level 1 up (8x8 -> 16x16)
        self.up1_res = ResnetBlock(c2 + c2, c2, self.time_emb_dim)
        self.up1_upsample = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(c2, c1, kernel_size=3, padding=1)
        ) # -> 16x16, 64 channels

        # Level 0 up (16x16) with self-attention
        self.up0_res = ResnetBlock(c1 + c1, c1, self.time_emb_dim)
        self.up0_attn = AttentionBlock(c1)

        # Output projection
        self.out_norm = nn.GroupNorm(min(32, c1), c1)
        self.out_act = nn.SiLU()
        self.conv_out = nn.Conv2d(c1, out_channels, kernel_size=3, padding=1)

    def forward(self, x, t, y):
        """
        x: (B, 4, 16, 16) noisy latent
        t: (B,) timesteps
        y: (B,) class labels in [0, 10]
        """
        # Embed timesteps and labels
        t_emb = self.time_mlp(t)
        y_emb = self.label_emb(y)
        cond = self.cond_proj(torch.cat([t_emb, y_emb], dim=-1))

        # Initial conv
        h = self.conv_in(x) # (B, 64, 16, 16)

        # Down Level 0
        h = self.down0_res1(h, cond)
        h = self.down0_attn(h)
        skip0 = h
        h = self.down0_down(h) # (B, 64, 8, 8)

        # Down Level 1
        h = self.down1_res1(h, cond)
        h = self.down1_res2(h, cond)
        skip1 = h
        h = self.down1_down(h) # (B, 128, 4, 4)

        # Down Level 2
        h = self.down2_res1(h, cond)
        h = self.down2_res2(h, cond)
        skip2 = h # (B, 256, 4, 4)

        # Bottleneck
        h = self.mid_res1(h, cond)
        h = self.mid_attn(h)
        h = self.mid_res2(h, cond) # (B, 256, 4, 4)

        # Up Level 2
        h = torch.cat([h, skip2], dim=1) # (B, 512, 4, 4)
        h = self.up2_res(h, cond)        # (B, 256, 4, 4)
        h = self.up2_upsample(h)         # (B, 128, 8, 8)

        # Up Level 1
        h = torch.cat([h, skip1], dim=1) # (B, 256, 8, 8)
        h = self.up1_res(h, cond)        # (B, 128, 8, 8)
        h = self.up1_upsample(h)         # (B, 64, 16, 16)

        # Up Level 0
        h = torch.cat([h, skip0], dim=1) # (B, 128, 16, 16)
        h = self.up0_res(h, cond)        # (B, 64, 16, 16)
        h = self.up0_attn(h)             # (B, 64, 16, 16)

        # Output
        out = self.conv_out(self.out_act(self.out_norm(h))) # (B, 4, 16, 16)
        return out

