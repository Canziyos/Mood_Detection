# ------------------------------------------------------------#
#   Grid search on lam_prefer_image & lam_entropy             #
# ------------------------------------------------------------#
import os, sys, torch, numpy as np, random, json
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.fusion.AV_Fusion import FusionAV
from dataloader import FlexibleFusionDataset, ConflictValDataset

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False

# Config.
use_latents = False
latent_dim = None
batch_size = 32
epochs = 150
patience = 15
gate_hidden = 32
oversample_audio  = False
frac_conflict = 0.3
lam_kl = 0.0
lr = 1e-3

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
num_classes = len(class_names)

results_dir = "./models/gridsearch_gate"
os.makedirs(results_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Sweep grid:
lam_prefer_image_grid = [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]
lam_entropy_grid = [0.0, 0.01, 0.05, 0.1]

summary = []

for lam_prefer_image in lam_prefer_image_grid:
    for lam_entropy in lam_entropy_grid:

        print(f"\n=== Training: lam_prefer_image={lam_prefer_image:.2f}, lam_entropy={lam_entropy:.2f} ===")

        # Datasets.
        train_ds = FlexibleFusionDataset(
            logits_audio_dir="./logits/audio/train",
            logits_image_dir="./logits/images/train",
            class_names=class_names,
            latent_audio_dir=None,
            latent_image_dir=None,
            pair_mode=False,
            oversample_audio=oversample_audio
        )
        val_ds = ConflictValDataset(
            logits_audio_dir="./logits/audio/val",
            logits_image_dir="./logits/images/val",
            class_names=class_names,
            latent_audio_dir=None,
            latent_image_dir=None,
            frac_conflict=frac_conflict
        )
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)

        # Model.
        fusion_head = FusionAV(
            num_classes=num_classes,
            fusion_mode="gate",
            latent_dim_audio=None,
            latent_dim_image=None,
            use_latents=False,
            gate_hidden=gate_hidden
        ).to(device)

        optimizer  = torch.optim.Adam(fusion_head.parameters(), lr=lr)
        scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        ce_criterion = torch.nn.CrossEntropyLoss()

        best_val, bad_epochs, best_state = float("inf"), 0, None
        ckpt = os.path.join(
            results_dir,
            f"gate_imgpref{lam_prefer_image:.2f}_ent{lam_entropy:.2f}.pth"
        )

        # train loop.
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
                    latent_audio=None, latent_image=None,
                    return_gate=True
                )
                ce = ce_criterion(logits_f, y)
                # Encourage more image gating (+ sign: Reward more image).
                prefer_image = alpha[:, 1].mean()
                entropy = -(alpha * (torch.log(alpha + 1e-8))).sum(dim=1).mean()
                loss = ce - lam_prefer_image * prefer_image - lam_entropy * entropy
                loss.backward()
                optimizer.step()
                running += loss.item() * y.size(0)

            train_loss = running / len(train_loader.dataset)
            scheduler.step()

            # Validare.
            fusion_head.eval()
            val_loss, correct, total, alpha_a_vals, alpha_i_vals = 0.0, 0, 0, [], []
            with torch.no_grad():
                for logits_a, logits_i, _, _, y in val_loader:
                    logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
                    logits_f, alpha = fusion_head.fuse_probs(
                        probs_audio=torch.softmax(logits_a, dim=1),
                        probs_image=torch.softmax(logits_i, dim=1),
                        pre_softmax_audio=logits_a, pre_softmax_image=logits_i,
                        latent_audio=None, latent_image=None,
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

            print(f"[Ep {ep:03}] "
                  f"train {train_loss:.4f} | val {val_loss:.4f} | acc {val_acc:.4f} "
                  f"| mean alpha_a {mean_alpha_a:.3f} | mean alpha_i {mean_alpha_i:.3f} | "
                  f"lr {scheduler.get_last_lr()[0]:.6f}")

            # early stop.
            if val_loss < best_val - 1e-5:
                best_val, bad_epochs = val_loss, 0
                best_state = fusion_head.state_dict()
                torch.save({"state_dict": best_state, "use_latents": use_latents}, ckpt)
            else:
                bad_epochs += 1
                if bad_epochs >= patience:
                    print("Early stopping.")
                    break

        # Load best for summary.
        if best_state:
            fusion_head.load_state_dict(best_state)
        summary.append(dict(
            lam_prefer_image=lam_prefer_image,
            lam_entropy=lam_entropy,
            val_loss=best_val,
            val_acc=val_acc,
            mean_alpha_a=mean_alpha_a,
            mean_alpha_i=mean_alpha_i,
            ckpt=ckpt
        ))

# Summary.
print("\n=== Grid Search Results ===")
summary.sort(key=lambda x: -x["val_acc"])  # sort by best accuracy.
for s in summary:
    print(json.dumps(s, indent=2))
with open(os.path.join(results_dir, "gridsearch_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)
