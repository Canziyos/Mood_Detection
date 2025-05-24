import numpy as np
import yaml
import os


def load_config(config_path="config.yaml"):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def compute_mean_std(folder):
    """Compute mean and std for all logits in a directory (e.g., training split)."""
    all_logits = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".npy"):
            arr = np.load(os.path.join(folder, fname))
            all_logits.append(arr.flatten())
    all_logits = np.concatenate(all_logits)
    mean = all_logits.mean()
    std = all_logits.std()
    return mean, std
