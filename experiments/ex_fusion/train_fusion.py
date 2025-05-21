import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
import sys
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from src.fusion.AV_Fusion import FusionAV
from dataloader import FusionPairDataset, HybridFusionPairDataset



# ==== EXPERIMENT SETTINGS (CHANGE THESE ONLY) ====
fusion_mode = "hybrid"   # "gate", "latent", or "hybrid"
latent_dim = 1280        # latent vector dim
num_classes = 6
batch_size = 32
epochs = 50
patience = 8

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "./models"
os.makedirs(results_dir, exist_ok=True)

# ========== DATA & MODEL SELECTION ==========
if fusion_mode == "gate":
    # Gate on logits only
    train_dataset = FusionPairDataset("./logits/audio/train", "./logits/images/train", class_names)
    val_dataset   = FusionPairDataset("./logits/audio/val",   "./logits/images/val",   class_names)
    fusion_head = FusionAV(num_classes=num_classes, fusion_mode="gate").to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_type = "gate"
elif fusion_mode == "latent":
    # Latent head on latents only
    train_dataset = FusionPairDataset("./latent/audio/train", "./latent/images/train", class_names)
    val_dataset   = FusionPairDataset("./latent/audio/val",   "./latent/images/val",   class_names)
    fusion_head = FusionAV(num_classes=num_classes, fusion_mode="latent",
                           latent_dim_audio=latent_dim, latent_dim_image=latent_dim).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_type = "latent"
elif fusion_mode == "hybrid":
    # Gate head on logits + latents
    train_dataset = HybridFusionPairDataset(
        logits_audio_dir="./logits/audio/train", logits_image_dir="./logits/images/train",
        latent_audio_dir="./latent/audio/train", latent_image_dir="./latent/images/train",
        class_names=class_names)
    val_dataset = HybridFusionPairDataset(
        logits_audio_dir="./logits/audio/val", logits_image_dir="./logits/images/val",
        latent_audio_dir="./latent/audio/val", latent_image_dir="./latent/images/val",
        class_names=class_names)
    fusion_head = FusionAV(num_classes=num_classes, fusion_mode="gate",
                           latent_dim_audio=latent_dim, latent_dim_image=latent_dim).to(device)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    train_type = "hybrid"
else:
    raise ValueError("fusion_mode must be 'gate', 'latent', or 'hybrid'.")

optimizer = torch.optim.Adam(fusion_head.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None
history = {"train_loss": [], "val_loss": [], "val_acc": []}

for epoch in range(epochs):
    fusion_head.train()
    total_loss = 0.0
    print(f"Epoch {epoch+1}/{epochs}: Training...")
    for batch_idx, batch in enumerate(train_loader):
        optimizer.zero_grad()
        if train_type == "gate":
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
        elif train_type == "latent":
            X_a, X_i, y = [t.to(device) for t in batch]
            logits = fusion_head.fuse_probs(
                probs_audio=None,
                probs_image=None,
                latent_audio=X_a,
                latent_image=X_i,
                return_logits=True
            )
        elif train_type == "hybrid":
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
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * (logits.shape[0])
        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
            print(f"  Training batch {batch_idx+1}/{len(train_loader)}")
    avg_loss = total_loss / len(train_loader.dataset)
    history["train_loss"].append(avg_loss)

    # Validation
    fusion_head.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    val_preds, val_trues = [], []
    print(f"Epoch {epoch+1}/{epochs}: Validating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loader):
            if train_type == "gate":
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
            elif train_type == "latent":
                X_a, X_i, y = [t.to(device) for t in batch]
                logits = fusion_head.fuse_probs(
                    probs_audio=None,
                    probs_image=None,
                    latent_audio=X_a,
                    latent_image=X_i,
                    return_logits=True
                )
            elif train_type == "hybrid":
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
            loss = criterion(logits, y)
            val_loss += loss.item() * (logits.shape[0])
            preds = torch.argmax(logits, dim=1)
            val_preds.extend(preds.cpu().numpy())
            val_trues.extend(y.cpu().numpy())
            correct += (preds == y).sum().item()
            total += y.size(0)
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(val_loader):
                print(f"  Validation batch {batch_idx+1}/{len(val_loader)}")
    val_loss = val_loss / len(val_loader.dataset)
    val_acc = correct / total
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{epochs}: Train loss {avg_loss:.4f} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f}")

    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = fusion_head.state_dict()
        epochs_no_improve = 0
        head_type = "gate" if train_type == "gate" else "latent" if train_type == "latent" else "gate_hybrid"
        torch.save(best_state, f"{results_dir}/best_{head_type}_head.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Stop at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break

if best_state is not None:
    fusion_head.load_state_dict(best_state)
    print("Best weights loaded.")

if head_type:
    print(f"Saved best weights to: {os.path.abspath(f'{results_dir}/best_{head_type}_head.pth')}")
