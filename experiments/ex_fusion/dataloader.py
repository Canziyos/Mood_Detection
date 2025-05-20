import numpy as np
import torch
from torch.utils.data import Dataset
import random

class FusionPairDataset(Dataset):
    def __init__(self, audio_dir, image_dir, class_names, mode="latent", seed=42):
        self.audio_data = []
        self.image_data = []
        self.labels = []
        self.class_names = class_names
        random.seed(seed)
        for idx, cname in enumerate(class_names):
            a_data = np.load(f"{audio_dir}/{cname}.npy")
            i_data = np.load(f"{image_dir}/{cname}.npy")
            n_a, n_i = len(a_data), len(i_data)
            n_max = max(n_a, n_i)
            for i in range(n_max):
                a = a_data[i % n_a]
                i_sample = i_data[random.randint(0, n_i-1)]
                self.audio_data.append(a)
                self.image_data.append(i_sample)
                self.labels.append(idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        a = torch.tensor(self.audio_data[idx], dtype=torch.float32)
        i = torch.tensor(self.image_data[idx], dtype=torch.float32)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return a, i, y
