import sys, os
sys.path.append(os.path.abspath('../src/fusion'))
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import LatentPairDataset
from AV_Fusion import FusionAV
from sklearn.metrics import classification_report, confusion_matrix

# === SETTINGS ===
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)
latent_dim_aud = 1280
latent_dim_img = 1280
batch_size = 64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "../results/latent_head"

# === Load TEST DATA ===
test_dataset = LatentPairDataset(
    audio_latent_dir="../latents/audio/test",
    image_latent_dir="../latents/Images/test",
    class_names=class_names
)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# === LOAD BEST FUSION HEAD ===
fusion_head = FusionAV(
    num_classes=num_classes,
    fusion_mode="latent",
    latent_dim_audio=latent_dim_aud,
    latent_dim_image=latent_dim_img,
    use_pre_softmax=False,
    mlp_on_latent=False
).to(device)
fusion_head.load_state_dict(torch.load(f"{results_dir}/best_latent_head.pth", map_location=device))
fusion_head.eval()

# === RUN TEST ===
test_preds, test_trues = [], []
print("Running test set evaluation...")
with torch.no_grad():
    for batch_idx, (X_a, X_i, y) in enumerate(test_loader):
        X_a, X_i, y = X_a.to(device), X_i.to(device), y.to(device)
        logits = fusion_head.fuse_probs(
            probs_audio=None, probs_image=None,
            latent_audio=X_a, latent_image=X_i,
            return_logits=True
        )
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_trues.extend(y.cpu().numpy())
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_loader):
            print(f"  Test batch {batch_idx+1}/{len(test_loader)}")

# === RESULTS ===
print("\nClassification report on test set (with best head loaded):")
print(classification_report(test_trues, test_preds, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(test_trues, test_preds))

# Optionally save predictions and ground truth
np.save(f"{results_dir}/test_preds.npy", np.array(test_preds))
np.save(f"{results_dir}/test_labels.npy", np.array(test_trues))

print("\nTest phase completed and results saved.")
