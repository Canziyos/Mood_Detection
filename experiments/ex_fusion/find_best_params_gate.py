"""
Grid-search trainer for the AudioImageFusion head (logits-only version, with mean-std normalization).
Any combination that crashes is skipped.
"""
import os, sys, time, json, itertools, torch, pandas as pd
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AudioImageFusion import AudioImageFusion
from dataloader import FlexibleFusionDataset, ConflictValDataset

print("\nGate...\n")

# ---------------------------------------------------------------------
# Logit normalization values (mean & std) for both modalities (from stats.py)
aud_logits_mean = -2.953
aud_logits_std  = 5.11
img_logits_mean = -0.592
img_logits_std  = 1.58

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "./models"
os.makedirs(results_dir, exist_ok=True)

# Grid search params.
search_space = {
    "oversample_audio": [False, True],
    "lam_kl": [0.0, 0.02, 0.1],
    "lr": [1e-4, 3e-4, 1e-3, 3e-3],
    "frac_conflict": [0.0, 0.3, 0.5, 0.7],
    "batch_size": [16, 32, 64],
}

grid = list(dict(zip(search_space, v)) for v in itertools.product(*search_space.values()))

def make_loader(split, *, cfg):
    batch_size = cfg.get("batch_size", 32)
    if split == "val":
        ds = ConflictValDataset(
            logits_audio_dir = f"../../logits/audio/{split}",
            logits_image_dir = f"../../logits/images/{split}",
            class_names = class_names,
            frac_conflict = cfg["frac_conflict"],
        )
        shuffle = False
    else:
        ds = FlexibleFusionDataset(
            logits_audio_dir = f"../../logits/audio/{split}",
            logits_image_dir = f"../../logits/images/{split}",
            class_names = class_names,
            pair_mode = (split != "train"),     # False = random, True = 1-to-1.
            oversample_audio = (split == "train") and cfg["oversample_audio"],
        )
        shuffle = (split == "train")
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)

def train_once(cfg):
    train_loader = make_loader("train", cfg=cfg)
    val_loader   = make_loader("val",   cfg=cfg)

    fusion = AudioImageFusion(
        num_classes=len(class_names),
        fusion_mode="gate"
    ).to(device)

    opt = torch.optim.Adam(fusion.parameters(), lr=cfg["lr"])
    ce_fn = torch.nn.CrossEntropyLoss()
    kl_fn = torch.nn.KLDivLoss(reduction="batchmean")

    best_val, best_acc, bad, start = float("inf"), 0.0, 0, time.time()
    for ep in range(1, 101):
        fusion.train()
        running = 0.0
        all_alpha_a, all_alpha_i = [], []

        for logits_a, logits_i, _, _, y in train_loader:
            logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
            # Mean-std normalize logits before gate.
            norm_logits_a = (logits_a - aud_logits_mean) / aud_logits_std
            norm_logits_i = (logits_i - img_logits_mean) / img_logits_std
            opt.zero_grad()
            fused, alpha_out = fusion.fuse_probs(
                torch.softmax(logits_a, 1), torch.softmax(logits_i, 1),
                norm_logits_a, norm_logits_i, return_gate=True
            )
            logits_f = fused

            alpha_a, alpha_i = alpha_out[:, 0], alpha_out[:, 1]
            all_alpha_a.append(alpha_a.detach().cpu())
            all_alpha_i.append(alpha_i.detach().cpu())

            ce = ce_fn(logits_f, y)
            pa, pi = torch.softmax(logits_a, 1), torch.softmax(logits_i, 1)
            kl = kl_fn(torch.log(pa + 1e-9), pi.detach()) + \
                 kl_fn(torch.log(pi + 1e-9), pa.detach())

            
            loss = ce + cfg["lam_kl"] * kl

            loss.backward()
            opt.step()
            running += loss.item() * y.size(0)

        train_loss = running / len(train_loader.dataset)
        mean_alpha_a = torch.cat(all_alpha_a).mean().item()
        mean_alpha_i = torch.cat(all_alpha_i).mean().item()

        # Validation.
        fusion.eval()
        vl, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for logits_a, logits_i, _, _, y in val_loader:
                logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
                # Mean-std normalize logits before gate.
                norm_logits_a = (logits_a - aud_logits_mean) / aud_logits_std
                norm_logits_i = (logits_i - img_logits_mean) / img_logits_std
                fused, alpha_out = fusion.fuse_probs(
                    torch.softmax(logits_a, 1), torch.softmax(logits_i, 1),
                    norm_logits_a, norm_logits_i, return_gate=True
                )
                logits_f = fused
                ce = ce_fn(logits_f, y)
                vl += ce.item() * y.size(0)
                corr += (torch.argmax(logits_f, 1) == y).sum().item()
                tot += y.size(0)
        vl /= len(val_loader.dataset)
        va = corr / tot

        print(f"[{ep:03}] train {train_loss:.4f} | val {vl:.4f} | acc {va:.4f} | "
              f"mean alpha_a {mean_alpha_a:.3f} | mean alpha_i {mean_alpha_i:.3f}",
              end="\r", flush=True)

        if vl < best_val - 1e-5:
            best_val, best_acc, bad = vl, va, 0
        else:
            bad += 1
            if bad >= 8: break
    print()
    return best_val, best_acc, time.time() - start

summary = []
for cfg in grid:
    tag = "_".join(f"{k}={v}" for k, v in cfg.items())
    print(f"\nRunning {tag}")
    try:
        vloss, vacc, sec = train_once(cfg)
        summary.append(dict(cfg, val_loss=vloss, val_acc=vacc, sec=sec))
    except Exception as e:
        print(f"Crash: {e}")

# Results.
print("\n=== grid summary ===")
summary.sort(key=lambda d: d["val_loss"])
for s in summary:
    print(json.dumps(s, indent=None))

# Save as JSON.
with open("_grid_fusion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save as CSV.
pd.DataFrame(summary).to_csv("grid_fusion_summary.csv", index=False)

# Print best config.
if summary:
    print("\nBest config by val_loss:")
    print(json.dumps(summary[0], indent=2))
