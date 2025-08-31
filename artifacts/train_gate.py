import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from utils.AudioImageFusion_bi import AudioImageFusion
from dataloader import FlexibleFusionDataset, ConflictValDataset
from utils.utils import load_config

config = load_config("config.yaml")

aud_logits_train_dir = config["logits"]["train_aud_logits_dir"]
img_logits_train_dir = config["logits"]["train_img_logits_dir"]
aud_logits_val_dir = config["logits"]["val_aud_logits_dir"]
img_logits_val_dir = config["logits"]["val_img_logits_dir"]
results_dir = config["models"]["root"]

training_cfg = config["training"]
batch_size = training_cfg["batch_size"]
epochs = training_cfg["epochs"]
patience = training_cfg["patience"]
lr = training_cfg["lr"]
oversample_audio  = training_cfg["oversample_audio"]
frac_conflict = training_cfg["frac_conflict"]
lam_kl = training_cfg.get("lam_kl", 0.0)
lam_entropy = training_cfg.get("lam_entropy", 0.0)

class_names = config["classes"]
num_classes = len(class_names)

norm_cfg = config["normalization"]
aud_logits_mean = norm_cfg["aud_logits_mean"]
aud_logits_std  = norm_cfg["aud_logits_std"]
img_logits_mean = norm_cfg["img_logits_mean"]
img_logits_std  = norm_cfg["img_logits_std"]

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_ds = FlexibleFusionDataset(
    logits_audio_dir=aud_logits_train_dir,
    logits_image_dir=img_logits_train_dir,
    class_names=class_names,
    pair_mode=False,
    oversample_audio=oversample_audio
)

val_ds = ConflictValDataset(
    logits_audio_dir=aud_logits_val_dir,
    logits_image_dir=img_logits_val_dir,
    class_names=class_names,
    frac_conflict=frac_conflict
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

fusion_head = AudioImageFusion(
    num_classes=num_classes,
    fusion_mode="gate",
).to(device)

optimizer = torch.optim.Adam(fusion_head.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
ce_criterion = torch.nn.CrossEntropyLoss()
kl_criterion = torch.nn.KLDivLoss(reduction="batchmean")

best_val, bad_epochs, best_state = float("inf"), 0, None
ckpt = os.path.join(results_dir, "gate_normalized_logits.pth")

def normalize_logits(logits, mean, std):
    mean = torch.tensor(mean, device=logits.device)
    std = torch.tensor(std, device=logits.device)
    return (logits - mean) / std

for ep in range(1, epochs + 1):
    fusion_head.train()
    running = 0.0
    train_entropy_sum, train_entropy_count = 0.0, 0

    for logits_a, logits_i, _, _, y in train_loader:
        logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
        norm_logits_a = normalize_logits(logits_a, aud_logits_mean, aud_logits_std)
        norm_logits_i = normalize_logits(logits_i, img_logits_mean, img_logits_std)

        optimizer.zero_grad()
        probs_audio = torch.softmax(logits_a, dim=1)
        probs_image = torch.softmax(logits_i, dim=1)
        logits_f, alpha = fusion_head.fuse_probs(
            probs_audio=probs_audio,
            probs_image=probs_image,
            pre_softmax_audio=norm_logits_a,
            pre_softmax_image=norm_logits_i,
            return_gate=True
        )
        ce = ce_criterion(logits_f, y)
        kl_loss = kl_criterion(torch.log(probs_audio + 1e-9), probs_image.detach()) + \
                  kl_criterion(torch.log(probs_image + 1e-9), probs_audio.detach())
        gate_entropy = -(alpha * (torch.log(alpha + 1e-8))).sum(dim=1).mean()
        loss = ce + lam_kl * kl_loss + lam_entropy * gate_entropy

        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)
        train_entropy_sum += gate_entropy.item() * y.size(0)
        train_entropy_count += y.size(0)

    train_loss = running / len(train_loader.dataset)
    train_entropy = train_entropy_sum / train_entropy_count
    scheduler.step()

    fusion_head.eval()
    val_loss, correct, total, alpha_a_vals, alpha_i_vals = 0.0, 0, 0, [], []
    val_entropy_sum, val_entropy_count = 0.0, 0
    with torch.no_grad():
        for logits_a, logits_i, _, _, y in val_loader:
            logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
            norm_logits_a = normalize_logits(logits_a, aud_logits_mean, aud_logits_std)
            norm_logits_i = normalize_logits(logits_i, img_logits_mean, img_logits_std)

            probs_audio = torch.softmax(logits_a, dim=1)
            probs_image = torch.softmax(logits_i, dim=1)
            logits_f, alpha = fusion_head.fuse_probs(
                probs_audio=probs_audio,
                probs_image=probs_image,
                pre_softmax_audio=norm_logits_a,
                pre_softmax_image=norm_logits_i,
                return_gate=True
            )
            val_loss += ce_criterion(logits_f, y).item() * y.size(0)
            correct += (logits_f.argmax(1) == y).sum().item()
            total += y.size(0)
            alpha_a_vals.extend(alpha[:, 0].detach().cpu().tolist())
            alpha_i_vals.extend(alpha[:, 1].detach().cpu().tolist())
            gate_entropy = -(alpha * (torch.log(alpha + 1e-8))).sum(dim=1).mean()
            val_entropy_sum += gate_entropy.item() * y.size(0)
            val_entropy_count += y.size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = correct / total
    mean_alpha_a = np.mean(alpha_a_vals) if alpha_a_vals else 0
    mean_alpha_i = np.mean(alpha_i_vals) if alpha_i_vals else 0
    val_entropy = val_entropy_sum / val_entropy_count

    print(
        f"Ep {ep:03}: "
        f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f} "
        f"| mean alpha_a {mean_alpha_a:.3f} | mean alpha_i {mean_alpha_i:.3f} "
        f"| train_entropy {train_entropy:.4f} | val_entropy {val_entropy:.4f} "
        f"| lr {scheduler.get_last_lr()[0]:.6f}"
    )

    if val_loss < best_val - 1e-5:
        best_val, bad_epochs = val_loss, 0
        best_state = fusion_head.state_dict()
        torch.save({"state_dict": best_state}, ckpt)
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

if best_state:
    fusion_head.load_state_dict(best_state)
    print(f"Best model re-loaded (val loss {best_val:.4f}, acc {val_acc:.4f}).")
