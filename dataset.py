import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
from vae_pipeline import get_fashion_mnist_transform, encode_to_latents

class LatentFashionMNISTDataset(Dataset):
    """
    Tensor dataset holding precomputed VAE latents (4x16x16) and class labels.
    """
    def __init__(self, latents, labels):
        self.latents = latents
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.latents[idx], self.labels[idx]

def get_raw_fashion_mnist_loaders(root="./data", batch_size=64):
    """
    Returns standard raw FashionMNIST DataLoaders (resized to 128x128x3).
    """
    transform = get_fashion_mnist_transform()
    train_dataset = FashionMNIST(root=root, train=True, download=False, transform=transform)
    test_dataset = FashionMNIST(root=root, train=False, download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    return train_loader, test_loader

def prepare_and_cache_latents(vae, root="./data", cache_dir=".", batch_size=64, device="cuda"):
    """
    Precomputes and caches VAE latents for train and test splits to accelerate
    diffusion training. Uses GPU interpolation for maximum throughput.
    """
    train_cache_path = os.path.join(cache_dir, "fashion_mnist_latents_train.pt")
    test_cache_path = os.path.join(cache_dir, "fashion_mnist_latents_test.pt")

    if os.path.exists(train_cache_path) and os.path.exists(test_cache_path):
        print(f"Loading cached latents from '{train_cache_path}' and '{test_cache_path}'...")
        train_data = torch.load(train_cache_path, map_location="cpu")
        test_data = torch.load(test_cache_path, map_location="cpu")
        return (
            LatentFashionMNISTDataset(train_data['latents'], train_data['labels']),
            LatentFashionMNISTDataset(test_data['latents'], test_data['labels'])
        )

    print("Precomputing VAE latents with GPU-accelerated pipeline (this runs once)...")
    train_dataset = FashionMNIST(root=root, train=True, download=False)
    test_dataset = FashionMNIST(root=root, train=False, download=False)

    def encode_split(dataset, split_name):
        raw_data = dataset.data  # (N, 28, 28) uint8
        raw_labels = dataset.targets # (N,) long
        N = len(dataset)
        latents_list = []

        for i in tqdm(range(0, N, batch_size), desc=f"Encoding {split_name}"):
            batch_data = raw_data[i:i+batch_size].unsqueeze(1).to(device).float() / 255.0
            batch_data = batch_data * 2.0 - 1.0  # normalize to [-1, 1]
            batch_data = batch_data.repeat(1, 3, 1, 1)  # 3 channels
            
            with torch.no_grad():
                # GPU bilinear interpolation to 128x128
                batch_128 = F.interpolate(batch_data, size=(128, 128), mode='bilinear', align_corners=False)
                # VAE encoding and scaling by s = 0.18215
                lats = encode_to_latents(vae, batch_128, sample=True)
                latents_list.append(lats.cpu())

        all_latents = torch.cat(latents_list, dim=0)
        return all_latents, raw_labels

    train_latents, train_labels = encode_split(train_dataset, "Train")
    test_latents, test_labels = encode_split(test_dataset, "Test")

    print(f"Saving cached latents: Train {train_latents.shape}, Test {test_latents.shape}")
    torch.save({'latents': train_latents, 'labels': train_labels}, train_cache_path)
    torch.save({'latents': test_latents, 'labels': test_labels}, test_cache_path)

    return (
        LatentFashionMNISTDataset(train_latents, train_labels),
        LatentFashionMNISTDataset(test_latents, test_labels)
    )

def get_latent_dataloaders(train_dataset, test_dataset, batch_size=128):
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    return train_loader, test_loader
