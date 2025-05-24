import os
import numpy as np
import matplotlib.pyplot as plt

# CONFIG.
audio_logits_dir = "../../logits/audio/train"
image_logits_dir = "../../logits/images/train"

# The means and stds found, calculated, previously.


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
audio_logits_all = np.concatenate(get_logits_list(audio_logits_dir))
image_logits_all = np.concatenate(get_logits_list(image_logits_dir))

# Normalized (mean-std) logits
audio_logits_norm = (audio_logits_all - AUDIO_MEAN) / AUDIO_STD
image_logits_norm = (image_logits_all - IMAGE_MEAN) / IMAGE_STD

# Summary stats
def print_stats(name, arr):
    print(f"{name} logits: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

print("=== Original ===")
print_stats("Audio", audio_logits_all)
print_stats("Image", image_logits_all)

print("\n=== Normalized ===")
print_stats("Audio (normalized)", audio_logits_norm)
print_stats("Image (normalized)", image_logits_norm)

# Plot histograms (side by side for before/after)
plt.figure(figsize=(14,6))

plt.subplot(1,2,1)
plt.hist(audio_logits_all, bins=50, alpha=0.5, label="Audio logits")
plt.hist(image_logits_all, bins=50, alpha=0.5, label="Image logits")
plt.legend()
plt.title("Original Distributions")
plt.xlabel("Logit value")
plt.ylabel("Frequency")

plt.subplot(1,2,2)
plt.hist(audio_logits_norm, bins=50, alpha=0.5, label="Audio logits (norm)")
plt.hist(image_logits_norm, bins=50, alpha=0.5, label="Image logits (norm)")
plt.legend()
plt.title("Normalized Distributions")
plt.xlabel("Normalized value")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()
