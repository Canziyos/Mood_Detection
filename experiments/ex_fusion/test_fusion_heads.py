import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from src.fusion.AV_Fusion import FusionAV
from sklearn.metrics import classification_report, confusion_matrix
from dataloader import FusionPairDataset, HybridFusionPairDataset

# Settings.
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)
batch_size = 32
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Select fusion.
fusion_mode = "hybrid"    # "gate", "latent", or "hybrid"
model_file = "./models/best_gate_hybrid_head.pth"  # change as needed

# Prepare model & data.
if fusion_mode == "gate":
    audio_dir = "./logits/audio/test"
    image_dir = "./logits/images/test"
    fusion_head = FusionAV(
        num_classes=num_classes,
        fusion_mode="gate",
        logits_dim_audio=6,
        logits_dim_image=6,
        use_pre_softmax=True
    ).to(device)
    test_dataset = FusionPairDataset(
        audio_dir=audio_dir,
        image_dir=image_dir,
        class_names=class_names
    )
    loader_mode = "logits"
elif fusion_mode == "latent":
    audio_dir = "./latent/audio/test"
    image_dir = "./latent/images/test"
    fusion_head = FusionAV(
        num_classes=num_classes,
        fusion_mode="latent",
        latent_dim_audio=1280,
        latent_dim_image=1280
    ).to(device)
    test_dataset = FusionPairDataset(
        audio_dir=audio_dir,
        image_dir=image_dir,
        class_names=class_names
    )
    loader_mode = "latent"
elif fusion_mode == "hybrid":
    # expects both logits and latent dirs.
    fusion_head = FusionAV(
        num_classes=num_classes,
        fusion_mode="gate",
        latent_dim_audio=1280,
        latent_dim_image=1280
    ).to(device)
    test_dataset = HybridFusionPairDataset(
        logits_audio_dir="./logits/audio/test",
        logits_image_dir="./logits/images/test",
        latent_audio_dir="./latent/audio/test",
        latent_image_dir="./latent/images/test",
        class_names=class_names
    )
    loader_mode = "hybrid"
else:
    raise ValueError("fusion_mode must be 'gate', 'latent', or 'hybrid'.")

fusion_head.load_state_dict(torch.load(model_file, map_location=device))
fusion_head.eval()
print(f"Loaded fusion head from: {model_file}")

test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Run tests
test_preds, test_trues = [], []

test_preds, test_trues = [], []
test_audio_preds, test_image_preds = [], []

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        if loader_mode == "hybrid":
            logits_a, logits_i, lat_a, lat_i, y = [t.to(device) for t in batch]
            softmax_a = torch.softmax(logits_a, dim=1)
            softmax_i = torch.softmax(logits_i, dim=1)
            logits = fusion_head.fuse_probs(
                probs_audio=softmax_a,
                probs_image=softmax_i,
                pre_softmax_audio=logits_a,
                pre_softmax_image=logits_i,
                latent_audio=lat_a,
                latent_image=lat_i,
                return_logits=True
            )
            audio_preds = torch.argmax(softmax_a, dim=1)
            image_preds = torch.argmax(softmax_i, dim=1)
        elif loader_mode == "logits":
            X_a, X_i, y = [t.to(device) for t in batch]
            softmax_a = torch.softmax(X_a, dim=1)
            softmax_i = torch.softmax(X_i, dim=1)
            logits = fusion_head.fuse_probs(
                probs_audio=softmax_a,
                probs_image=softmax_i,
                pre_softmax_audio=X_a,
                pre_softmax_image=X_i,
                return_logits=True
            )
            audio_preds = torch.argmax(softmax_a, dim=1)
            image_preds = torch.argmax(softmax_i, dim=1)
        elif loader_mode == "latent":
            X_a, X_i, y = [t.to(device) for t in batch]
            logits = fusion_head.fuse_probs(
                probs_audio=None,
                probs_image=None,
                latent_audio=X_a,
                latent_image=X_i,
                return_logits=True
            )
            audio_preds = torch.argmax(logits, dim=1)
            image_preds = torch.argmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        test_preds.extend(preds.cpu().numpy())
        test_trues.extend(y.cpu().numpy())
        test_audio_preds.extend(audio_preds.cpu().numpy())
        test_image_preds.extend(image_preds.cpu().numpy())

# === RESULTS === #
print(f"\nClassification report on test set ({fusion_mode} head):")
print(classification_report(test_trues, test_preds, target_names=class_names))
print("\nConfusion Matrix:")
print(confusion_matrix(test_trues, test_preds))



df = pd.DataFrame({
    "class": [class_names[y] for y in test_trues],
    "fusion_pred": [class_names[y] for y in test_preds],
    "audio_pred": [class_names[y] for y in test_audio_preds],   # If you get these
    "image_pred": [class_names[y] for y in test_image_preds]    # If you get these
})
df.to_csv(f"./results/sync_fusion_results_{fusion_mode}.csv", index=False)
print("\nTest phase completed and results saved.")