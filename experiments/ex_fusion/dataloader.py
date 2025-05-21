import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from src.fusion.AV_Fusion import FusionAV


# ========= BASIC DATASET ==========
class FusionPairDataset(Dataset):
    def __init__(self, audio_dir, image_dir, class_names, seed=42):
        self.audio_data = []
        self.image_data = []
        self.labels = []
        np.random.seed(seed)
        for idx, cname in enumerate(class_names):
            a_data = np.load(f"{audio_dir}/{cname}.npy")
            i_data = np.load(f"{image_dir}/{cname}.npy")
            n = min(len(a_data), len(i_data))
            for i in range(n):
                self.audio_data.append(a_data[i])
                self.image_data.append(i_data[i])
                self.labels.append(idx)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        a = torch.tensor(self.audio_data[idx], dtype=torch.float32)
        i = torch.tensor(self.image_data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return a, i, y

# ========= HYBRID DATASET ==========
class HybridFusionPairDataset(Dataset):
    def __init__(self, logits_audio_dir, logits_image_dir, latent_audio_dir, latent_image_dir, class_names, seed=42):
        self.logits_audio_data = []
        self.logits_image_data = []
        self.latent_audio_data = []
        self.latent_image_data = []
        self.labels = []
        np.random.seed(seed)
        for idx, cname in enumerate(class_names):
            la = np.load(f"{latent_audio_dir}/{cname}.npy")
            li = np.load(f"{latent_image_dir}/{cname}.npy")
            ga = np.load(f"{logits_audio_dir}/{cname}.npy")
            gi = np.load(f"{logits_image_dir}/{cname}.npy")
            n = min(len(la), len(li), len(ga), len(gi))
            for i in range(n):
                self.logits_audio_data.append(ga[i])
                self.logits_image_data.append(gi[i])
                self.latent_audio_data.append(la[i])
                self.latent_image_data.append(li[i])
                self.labels.append(idx)
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        logits_a = torch.tensor(self.logits_audio_data[idx], dtype=torch.float32)
        logits_i = torch.tensor(self.logits_image_data[idx], dtype=torch.float32)
        latent_a = torch.tensor(self.latent_audio_data[idx], dtype=torch.float32)
        latent_i = torch.tensor(self.latent_image_data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return logits_a, logits_i, latent_a, latent_i, y
