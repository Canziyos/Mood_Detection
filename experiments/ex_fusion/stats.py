import os
import numpy as np
import matplotlib.pyplot as plt

# CONFIG.
audio_logits_dir = "../../logits/audio/train"
image_logits_dir = "../../logits/images/train"

# Helper to load logits.
def get_logits_list(folder):
    logits_list = []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".npy"):
            path = os.path.join(folder, fname)
            arr = np.load(path)
            logits_list.append(arr.flatten())
    return logits_list

# Load all logits.
audio_logits_all = get_logits_list(audio_logits_dir)
image_logits_all = get_logits_list(image_logits_dir)

audio_logits_all = np.concatenate(audio_logits_all)
image_logits_all = np.concatenate(image_logits_all)

# summary stats.
def print_stats(name, arr):
    print(f"{name} logits: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

print_stats("Audio", audio_logits_all)
print_stats("Image", image_logits_all)

# Plot histograms.
plt.figure(figsize=(10,5))
plt.hist(audio_logits_all, bins=50, alpha=0.5, label="Audio logits")
plt.hist(image_logits_all, bins=50, alpha=0.5, label="Image logits")
plt.legend()
plt.title("Distribution of Audio and Image Logits")
plt.xlabel("Logit value")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
