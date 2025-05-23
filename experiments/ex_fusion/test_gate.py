import os, sys, torch, pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np

# Repo path hack.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fusion.AV_Fusion import FusionAV
from dataloader import FlexibleFusionDataset

# Use best params found in grid search.
use_latents      = False
latent_dim       = 1280  # ignored since use_latents=False.
gate_hidden      = 32    # FROM grid search
ckpt_path        = "./models/best_gate_head_logits.pth"

# Static config.
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
batch_size  = 64
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset.
test_ds = FlexibleFusionDataset(
    logits_audio_dir="./logits/audio/test",
    logits_image_dir="./logits/images/test",
    latent_audio_dir=None,
    latent_image_dir=None,
    class_names=class_names,
    pair_mode=True        # 1-to-1 pairs for deterministic metrics.
)
test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

# Model.
ckpt = torch.load(ckpt_path, map_location="cpu")
fusion_head = FusionAV(
    num_classes=len(class_names),
    fusion_mode="gate",
    latent_dim_audio=None,
    latent_dim_image=None,
    use_latents=False,
    gate_hidden=gate_hidden
).to(device)
fusion_head.load_state_dict(ckpt["state_dict"])
fusion_head.eval()
print(f"Loaded checkpoint: {ckpt_path} (use_latents={use_latents})  [gate_hidden={gate_hidden}]")

# Inference.
y_true, y_pred = [], []
a_pred, i_pred = [], []
alpha_a_list, alpha_i_list = [], []
sample_classes = []

with torch.no_grad():
    for logits_a, logits_i, lat_a, lat_i, y in test_loader:
        logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)

        fused, alpha_out = fusion_head.fuse_probs(
            probs_audio=torch.softmax(logits_a, dim=1),
            probs_image=torch.softmax(logits_i, dim=1),
            pre_softmax_audio=logits_a, pre_softmax_image=logits_i,
            latent_audio=None, latent_image=None,
            return_gate=True
        )

        # shape: (B,2).
        if alpha_out.shape[1] == 1:
            alpha_a = alpha_out.squeeze(1)
            alpha_i = 1.0 - alpha_a
        else:
            alpha_a, alpha_i = alpha_out[:, 0], alpha_out[:, 1]

        y_true.extend(y.cpu().numpy())
        y_pred.extend(fused.argmax(1).cpu().numpy())
        a_pred.extend(logits_a.argmax(1).cpu().numpy())
        i_pred.extend(logits_i.argmax(1).cpu().numpy())
        alpha_a_list.extend(alpha_a.cpu().numpy())
        alpha_i_list.extend(alpha_i.cpu().numpy())
        sample_classes.extend(y.cpu().numpy())

# Metrics.
print("\n=== Final Gate Fusion Model Test Report ===\n")
print(classification_report(y_true, y_pred, target_names=class_names, digits=3))
print("\nConfusion matrix:")
print(confusion_matrix(y_true, y_pred))

print(f"\nOverall alpha audio (mean):  {np.mean(alpha_a_list):.3f}   |  alpha image (mean): {np.mean(alpha_i_list):.3f}")

# Per-class alpha (gate weight) stats.
print("\nPer-class mean gate weights (Alpha):")
for i, name in enumerate(class_names):
    mask = np.array(sample_classes) == i
    if np.sum(mask) == 0:
        continue
    print(f"  {name:>8} | Alfa_audio: {np.mean(np.array(alpha_a_list)[mask]):.3f}  | Alfa_image: {np.mean(np.array(alpha_i_list)[mask]):.3f}")

# Per-class accuracy.
print("\nPer-class accuracy:")
for i, name in enumerate(class_names):
    class_mask = np.array(y_true) == i
    class_acc = accuracy_score(np.array(y_true)[class_mask], np.array(y_pred)[class_mask])
    print(f"  {name:>8} : {class_acc:.3f}")

# Save results.
os.makedirs("./results", exist_ok=True)
suffix = "lat" if use_latents else "logits"
csv_path = f"./results/sync_fusion_results_{suffix}_final.csv"
pd.DataFrame({
    "class"       : [class_names[i] for i in y_true],
    "fusion_pred" : [class_names[i] for i in y_pred],
    "audio_pred"  : [class_names[i] for i in a_pred],
    "image_pred"  : [class_names[i] for i in i_pred],
    "alpha_audio" : alpha_a_list,
    "alpha_image" : alpha_i_list
}).to_csv(csv_path, index=False)
print(f"\n Test finished — results saved to {csv_path}")
print(f"\nOverall accuracy: {accuracy_score(y_true, y_pred):.3f}")
