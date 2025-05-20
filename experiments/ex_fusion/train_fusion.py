import sys, os
sys.path.append(os.path.abspath('../src/fusion'))
sys.path.append(os.path.abspath('.'))

import numpy as np
import torch
from torch.utils.data import DataLoader
from dataloader import FusionPairDataset
from AV_Fusion import FusionAV
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# Settings
class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)
latent_dim_aud = 1280
latent_dim_img = 1280
epochs = 25
batch_size = 64
patience = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "../results/latent_head"
os.makedirs(results_dir, exist_ok=True)

#print(np.load("../latents/audio/train/Angry.npy").shape)
# print(np.load("../latents/Images/train/Angry.npy").shape)

# Data
train_dataset = FusionPairDataset(
    audio_latent_dir="../latents/audio/train",
    image_latent_dir="../latents/Images/train",
    class_names=class_names
)
val_dataset = FusionPairDataset(
    audio_latent_dir="../latents/audio/val",
    image_latent_dir="../latents/Images/val",
    class_names=class_names
)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Fusion Head.
fusion_head = FusionAV(
    num_classes=num_classes,
    fusion_mode="latent",
    latent_dim_audio=latent_dim_aud,
    latent_dim_image=latent_dim_img,
    use_pre_softmax=False,
    mlp_on_latent=False
).to(device)

optimizer = torch.optim.Adam(fusion_head.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

# Early Stopping and logging.
best_val_loss = float("inf")
epochs_no_improve = 0
best_state = None
history = {"train_loss": [], "val_loss": [], "val_acc": []}
all_val_preds = []
all_val_labels = []
# Training & validation loop (NO test phase here)
for epoch in range(epochs):
    fusion_head.train()
    total_loss = 0.0
    print(f"Epoch {epoch+1}/{epochs}: Training...")
    for batch_idx, (X_a, X_i, y) in enumerate(train_loader):
        X_a, X_i, y = X_a.to(device), X_i.to(device), y.to(device)
        optimizer.zero_grad()
        logits = fusion_head.fuse_probs(
            probs_audio=None, probs_image=None,
            latent_audio=X_a, latent_image=X_i,
            return_logits=True
        )
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_a.size(0)
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
        for batch_idx, (X_a, X_i, y) in enumerate(val_loader):
            X_a, X_i, y = X_a.to(device), X_i.to(device), y.to(device)
            logits = fusion_head.fuse_probs(
                probs_audio=None, probs_image=None,
                latent_audio=X_a, latent_image=X_i,
                return_logits=True
            )
            loss = criterion(logits, y)
            val_loss += loss.item() * X_a.size(0)
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
    all_val_preds = val_preds
    all_val_labels = val_trues

    print(f"Epoch {epoch+1}/{epochs}: Train loss {avg_loss:.4f} | Val loss {val_loss:.4f} | Val acc {val_acc:.4f}")

    # Early stopping check.
    if val_loss < best_val_loss - 1e-5:
        best_val_loss = val_loss
        best_state = fusion_head.state_dict()
        epochs_no_improve = 0
        torch.save(best_state, f"{results_dir}/best_latent_head.pth")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= patience:
            print(f"Stop at epoch {epoch+1}. Best val loss: {best_val_loss:.4f}")
            break

# Restore best weights.
if best_state is not None:
    fusion_head.load_state_dict(best_state)
    print("Best weights loaded.")
