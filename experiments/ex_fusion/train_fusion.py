import os, sys, torch, numpy as np, random
from torch.utils.data import DataLoader

# Repo path hack.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AudioImageFusion import AudioImageFusion
from dataloader import FlexibleFusionDataset, ConflictValDataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Params
batch_size        = 32
epochs            = 150
patience          = 15
oversample_audio  = False
frac_conflict     = 0.3
lam_entropy       = 0.01
lam_prefer_image  = 0.05
lr                = 1e-3

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)

results_dir = "../../models"
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets
train_ds = FlexibleFusionDataset(
    logits_audio_dir="../../logits/audio/train",
    logits_image_dir="../../logits/images/train",
    class_names=class_names,
    pair_mode=False,
    oversample_audio=oversample_audio
)

val_ds = ConflictValDataset(
    logits_audio_dir="../../logits/audio/val",
    logits_image_dir="../../logits/images/val",
    class_names=class_names,
    frac_conflict=frac_conflict
)

train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

# MODEL.
fusion_head = AudioImageFusion(
    num_classes=num_classes,
    fusion_mode="gate",
).to(device)

# OPTIM / LOSSES.
optimizer  = torch.optim.Adam(fusion_head.parameters(), lr=lr)
scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
ce_criterion = torch.nn.CrossEntropyLoss()

best_val, bad_epochs, best_state = float("inf"), 0, None
ckpt = f"{results_dir}/best_gate_head_logits.pth"
for ep in range(1, epochs + 1):
    fusion_head.train()
    running = 0.0

    for logits_a, logits_i, _, _, y in train_loader:
        logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)

        optimizer.zero_grad()
        logits_f, alpha = fusion_head.fuse_probs(
            probs_audio=torch.softmax(logits_a, dim=1),
            probs_image=torch.softmax(logits_i, dim=1),
            pre_softmax_audio=logits_a, pre_softmax_image=logits_i,
            return_gate=True
        )
        ce = ce_criterion(logits_f, y)
        prefer_image = alpha[:, 1].mean()
        gate_entropy = -(alpha * (torch.log(alpha + 1e-8))).sum(dim=1).mean()
        loss = ce + lam_prefer_image * prefer_image + lam_entropy * gate_entropy

        loss.backward()
        optimizer.step()
        running += loss.item() * y.size(0)

    train_loss = running / len(train_loader.dataset)
    scheduler.step()

    # Validate.
    fusion_head.eval()
    val_loss, correct, total, alpha_a_vals, alpha_i_vals = 0.0, 0, 0, [], []
    with torch.no_grad():
        for logits_a, logits_i, _, _, y in val_loader:
            logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
            logits_f, alpha = fusion_head.fuse_probs(
                probs_audio=torch.softmax(logits_a, dim=1),
                probs_image=torch.softmax(logits_i, dim=1),
                pre_softmax_audio=logits_a, pre_softmax_image=logits_i,
                return_gate=True
            )
            val_loss += ce_criterion(logits_f, y).item() * y.size(0)
            correct  += (logits_f.argmax(1) == y).sum().item()
            total    += y.size(0)
            alpha_a_vals.extend(alpha[:, 0].detach().cpu().tolist())
            alpha_i_vals.extend(alpha[:, 1].detach().cpu().tolist())

    val_loss /= len(val_loader.dataset)
    val_acc   = correct / total
    mean_alpha_a = np.mean(alpha_a_vals) if alpha_a_vals else 0
    mean_alpha_i = np.mean(alpha_i_vals) if alpha_i_vals else 0

    print(f"Ep {ep:03}: "
          f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f} "
          f"| mean alpha_a {mean_alpha_a:.3f} | mean alpha_i {mean_alpha_i:.3f} | lr {scheduler.get_last_lr()[0]:.6f}")

    # Early stop.
    if val_loss < best_val - 1e-5:
        best_val, bad_epochs = val_loss, 0
        best_state = fusion_head.state_dict()
        torch.save({"state_dict": best_state}, ckpt)
    else:
        bad_epochs += 1
        if bad_epochs >= patience:
            print("Early stopping.")
            break

# Load best.
if best_state:
    fusion_head.load_state_dict(best_state)
    print(f"Best model re-loaded (val loss {best_val:.4f}, acc {val_acc:.4f}).")
