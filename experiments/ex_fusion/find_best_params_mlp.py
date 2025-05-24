
"""
Grid-search trainer for the FusionAV head.
----------------------------------------
Any combination that crashes is skipped.
"""
import os, sys, time, json, itertools, torch, pandas as pd
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AudioImageFusion import AudioImageFusion
from dataloader import FlexibleFusionDataset, ConflictValDataset

# --------------------------------------------------------------#

class_names = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
device      = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = "../../models"; os.makedirs(results_dir, exist_ok=True)

# Extended grid: more lam_kl, add lam_entropy.
search_space = {
    "use_latents" : [False],                    # Only logits.
    "oversample_audio" : [False, True],
    "gate_hidden" : [32, 64],
    "lam_kl" : [0.0, 0.02, 0.1],           # Tried higher KLs.
    "lam_entropy" : [0.0, 0.01, 0.05, 0.1],     # Entropy penalty grid.
    "lr" : [1e-3],
    "frac_conflict" : [0.3, 0.5],
}

grid = list(dict(zip(search_space, v)) for v in itertools.product(*search_space.values()))



def make_loader(split, *, cfg):
    if split == "val":
        ds = ConflictValDataset(
            logits_audio_dir = f"./logits/audio/{split}",
            logits_image_dir = f"./logits/images/{split}",
            class_names = class_names,
            latent_audio_dir = f"./latent/audio/{split}"  if cfg["use_latents"] else None,
            latent_image_dir = f"./latent/images/{split}" if cfg["use_latents"] else None,
            frac_conflict    = cfg["frac_conflict"],
        )
        shuffle = False
    else:
        ds = FlexibleFusionDataset(
            logits_audio_dir = f"./logits/audio/{split}",
            logits_image_dir = f"./logits/images/{split}",
            class_names = class_names,
            latent_audio_dir = f"./latent/audio/{split}"  if cfg["use_latents"] else None,
            latent_image_dir = f"./latent/images/{split}" if cfg["use_latents"] else None,
            pair_mode = (split != "train"),     # False = random, True = 1-to-1
            oversample_audio = (split == "train") and cfg["oversample_audio"],
        )
        shuffle = (split == "train")

    return DataLoader(ds, batch_size=32, shuffle=shuffle)

def train_once(cfg):
    train_loader = make_loader("train", cfg=cfg)
    val_loader   = make_loader("val",   cfg=cfg)

    fusion = AudioImageFusion(
        num_classes = len(class_names),
        fusion_mode = "gate",
        latent_dim_audio = 1280 if cfg["use_latents"] else None,
        latent_dim_image = 1280 if cfg["use_latents"] else None,
        use_latents = cfg["use_latents"],
        gate_hidden= cfg["gate_hidden"]
    ).to(device)

    opt = torch.optim.Adam(fusion.parameters(), lr=cfg["lr"])
    ce_fn = torch.nn.CrossEntropyLoss()
    kl_fn = torch.nn.KLDivLoss(reduction="batchmean")

    best_val, best_acc, bad, start = float("inf"), 0.0, 0, time.time()
    for ep in range(1, 101):
        fusion.train(); running = 0.0
        all_alpha_a, all_alpha_i = [], []

        for logits_a, logits_i, lat_a, lat_i, y in train_loader:
            logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
            lat_a = lat_a.to(device) if cfg["use_latents"] else None
            lat_i = lat_i.to(device) if cfg["use_latents"] else None

            opt.zero_grad()
            # return_gate=True to get alphas.
            fused, alpha_out = fusion.fuse_probs(
                torch.softmax(logits_a, 1), torch.softmax(logits_i, 1),
                logits_a, logits_i, lat_a, lat_i, return_gate=True
            )
            logits_f = fused

            # Gate alpha processing.
            if alpha_out.shape[1] == 1:
                alpha_a = alpha_out.squeeze(1)
                alpha_i = 1.0 - alpha_a
                alpha_softmax = torch.stack([alpha_a, alpha_i], dim=1)
            else:
                alpha_a, alpha_i = alpha_out[:, 0], alpha_out[:, 1]
                alpha_softmax = alpha_out

            all_alpha_a.append(alpha_softmax[:,0].detach().cpu())
            all_alpha_i.append(alpha_softmax[:,1].detach().cpu())

            ce = ce_fn(logits_f, y)
            pa, pi = torch.softmax(logits_a,1), torch.softmax(logits_i,1)
            kl = kl_fn(torch.log(pa+1e-9), pi.detach()) + \
                 kl_fn(torch.log(pi+1e-9), pa.detach())

            entropy = - (alpha_softmax * torch.log(alpha_softmax + 1e-8)).sum(dim=1).mean()
            loss = ce + cfg["lam_kl"] * kl - cfg["lam_entropy"] * entropy  # maximize entropy.

            loss.backward(); opt.step()
            running += loss.item()*y.size(0)

        train_loss = running/len(train_loader.dataset)
        mean_alpha_a = torch.cat(all_alpha_a).mean().item()
        mean_alpha_i = torch.cat(all_alpha_i).mean().item()

        scheduler = None  # removed scheduler for grid search clarity.
        # Validation.
        fusion.eval(); vl, corr, tot = 0.0, 0, 0
        with torch.no_grad():
            for logits_a, logits_i, lat_a, lat_i, y in val_loader:
                logits_a, logits_i, y = logits_a.to(device), logits_i.to(device), y.to(device)
                lat_a = lat_a.to(device) if cfg["use_latents"] else None
                lat_i = lat_i.to(device) if cfg["use_latents"] else None
                fused, alpha_out = fusion.fuse_probs(
                    torch.softmax(logits_a,1), torch.softmax(logits_i,1),
                    logits_a, logits_i, lat_a, lat_i, return_gate=True
                )
                logits_f = fused
                ce = ce_fn(logits_f, y)
                vl += ce.item()*y.size(0)
                corr += (torch.argmax(logits_f,1)==y).sum().item()
                tot  += y.size(0)
        vl /= len(val_loader.dataset); va = corr/tot

        print(f"[{ep:03}] train {train_loss:.4f} | val {vl:.4f} | acc {va:.4f} | "
              f"mean alpha_a {mean_alpha_a:.3f} | mean alpha_i {mean_alpha_i:.3f}",
              end="\r", flush=True)

        if vl < best_val-1e-5:
            best_val, best_acc, bad = vl, va, 0
        else:
            bad += 1
            if bad >= 8: break
    print()
    return best_val, best_acc, time.time()-start

# main loop #
summary = []
for cfg in grid:
    tag = "_".join(f"{k}={v}" for k,v in cfg.items())
    print(f"\nRunning {tag}")
    try:
        vloss, vacc, sec = train_once(cfg)
        summary.append(dict(cfg, val_loss=vloss, val_acc=vacc, sec=sec))
    except Exception as e:
        print(f"Crash: {e}")

# results.
print("\n=== grid summary ===")
summary.sort(key=lambda d: d["val_loss"])
for s in summary:
    print(json.dumps(s, indent=None))
# Save as JSON.
with open("grid_mlp_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

# Save as CSV.
pd.DataFrame(summary).to_csv("grid_mlp_summary.csv", index=False)

# Print best config.
if summary:
    print("\nBest config by val_loss:")
    print(json.dumps(summary[0], indent=2))
