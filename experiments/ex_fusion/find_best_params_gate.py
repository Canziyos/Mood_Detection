import os, sys, time, json, itertools, torch, pandas as pd
from torch.utils.data import DataLoader

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from AudioImageFusion import AudioImageFusion
from dataloader import FlexibleFusionDataset, ConflictValDataset
from utils import load_config


# ---------------------------------------------------------------------
# Load YAML config for all static params.
config = load_config("config.yaml")

# Normalization values.
norm_cfg = config["normalization"]
aud_logits_mean = norm_cfg["aud_logits_mean"]
aud_logits_std  = norm_cfg["aud_logits_std"]
img_logits_mean = norm_cfg["img_logits_mean"]
img_logits_std  = norm_cfg["img_logits_std"]

# Paths from config.
logits_train_audio = config["logits"]["train_aud_logits_dir"]
logits_train_image = config["logits"]["train_img_logits_dir"]
logits_val_audio   = config["logits"]["val_aud_logits_dir"]
logits_val_image   = config["logits"]["val_img_logits_dir"]

class_names = config["classes"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

results_dir = config["results_dir"]["root"]
os.makedirs(results_dir, exist_ok=True)

# Grid search params (NOT from YAML).
search_space = {
    "batch_size": [32, 64],
    "oversample_audio": [False],
    "frac_conflict": [0.0, 0.3],
    "lam_kl": [0.0, 0.02, 0.05],
    "lam_entropy": [0.0, 0.01, 0.05],
    "lr": [0.0001, 0.0003, 0.001],
}

grid = list(dict(zip(search_space, v)) for v in itertools.product(*search_space.values()))

def make_loader(split, *, cfg):
    batch_size = cfg.get("batch_size", 32)
    if split == "val":
        ds = ConflictValDataset(
            logits_audio_dir = logits_val_audio,
            logits_image_dir = logits_val_image,
            class_names = class_names,
            frac_conflict = cfg["frac_conflict"],
        )
        shuffle = False
    else:
        ds = FlexibleFusionDataset(
            logits_audio_dir = logits_train_audio,
            logits_image_dir = logits_train_image,
            class_names = class_names,
            pair_mode = False,
            oversample_audio = cfg["oversample_audio"],
        )
        shuffle = True
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
            gate_entropy = -(alpha_out * (torch.log(alpha_out + 1e-8))).sum(dim=1).mean()

            loss = ce + cfg["lam_kl"] * kl + cfg["lam_entropy"] * gate_entropy

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


print("\n=== grid summary ===")
summary.sort(key=lambda d: d["val_loss"])
for s in summary:
    print(json.dumps(s, indent=None))

with open("finetune2_fusion_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

if summary:
    print("\nBest config by val_loss:")
    print(json.dumps(summary[0], indent=2))
