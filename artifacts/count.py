import os
import numpy as np
from pathlib import Path

# Config.
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
branches = ["audio", "images"]
splits = ["train", "val", "test"]

# Paths to raw files.
raw_data_base = "./dataset"
npy_base = "./logits"

for split in splits:
    print(f"\n=== {split.upper()} ===")
    for branch in branches:
        print(f"\n{branch.capitalize()} samples:")
        for cls in class_names:
            # Count in .npy file.
            npy_path = os.path.join(npy_base, branch, split, f"{cls}.npy")
            npy_count = "MISSING"
            if os.path.exists(npy_path):
                data = np.load(npy_path)
                npy_count = len(data)

            # Count raw files.
            raw_dir = Path(raw_data_base) / branch / split / cls
            if branch == "audio":
                file_types = ['*.wav']
            else:
                file_types = ['*.jpg', '*.jpeg', '*.png']
            raw_files = []
            for pattern in file_types:
                raw_files.extend(raw_dir.glob(pattern))
            raw_count = len(raw_files) if raw_dir.exists() else "MISSING"

            print(f"  {cls:<8} : npy={npy_count:>5} | files={raw_count}")

