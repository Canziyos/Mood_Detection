import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.utils import load_config
from experiments.ex_fusion.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_fusion.image_model_interface import load_image_model, extract_image_features
from utils.AudioImageFusion_bi import AudioImageFusion

config = load_config("config.yaml")

# Normalization values (from config).
aud_logits_mean = config["normalization"]["aud_logits_mean"]
aud_logits_std = config["normalization"]["aud_logits_std"]
img_logits_mean = config["normalization"]["img_logits_mean"]
img_logits_std = config["normalization"]["img_logits_std"]

class_names = config["classes"]

# Model paths: these point to the best/final models trained on our main data split.
audio_model_path = config["models"]["audio_model"]
image_model_path = config["models"]["image_model"]
ckpt_path = config["models"]["gate"]

# If you don't want to test models trained on CREMA-D, do not uncomment or use these paths :😏
# image_model_path = config["models"]["img_trained_on_crema"]
# audio_model_path = config["models"]["aud_trained_on_crema"]

# By default, use our standard held-out test split.
audio_root = config["data"]["aud_test_dir"]
image_root = config["data"]["img_test_dir"]

# If you want to test on external or cross-dataset test sets (emo_db audio, raf-db images etc.),
# uncomment and use these lines instead:
# audio_root = config["data"]["emo_db_test"]
# image_root = config["data"]["raf_db_test"]

# Set exp_name to tag/label this run for later identification.
# This name will be included in the output CSV file name,
# so we can match each CSV with the experiment setup ('regular', 'corpos', 'crema', etc.).
# exp_name = "regular"
exp_name = "our_cross_gated"
# exp_name = "crema_cross_gated"

out_dir = config["results_dir"]["root"]

# Fusion_type.
fusion_type = "gate"   # This script, now, is for gate only.

os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/fusion_{exp_name}_{fusion_type}.csv"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Audio and image backbones loaded.")

audio_model, _ = load_audio_model(model_path=audio_model_path)
image_model = load_image_model(model_path=image_model_path)

# Load gate fusion model.
ckpt = torch.load(ckpt_path, map_location="cpu")
fusion_head = AudioImageFusion(
    num_classes=len(class_names),
    fusion_mode="gate"
).to(device)
fusion_head.load_state_dict(ckpt["state_dict"])
fusion_head.eval()
print("Gate head loaded.")

def normalize_logits(logits, mean, std):
    return (logits - mean) / std

records, y_true, y_a, y_i, y_f = [], [], [], [], []
alpha_a_list, alpha_i_list = [], []

with torch.no_grad():
    for cname in class_names:
        a_dir = os.path.join(audio_root, cname)
        i_dir = os.path.join(image_root, cname)
        if not os.path.exists(a_dir) or not os.path.exists(i_dir):
            print(f"Skipping class '{cname}' (missing folder).")
            continue

        a_files = sorted([f for f in os.listdir(a_dir) if f.endswith(".wav")])
        i_files = sorted([f for f in os.listdir(i_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        N = min(len(a_files), len(i_files))
        if N == 0:
            print(f"Skipping class '{cname}' (no paired files).")
            continue

        for idx in range(N):
            # Audio forward.
            wav, sr = torchaudio.load(os.path.join(a_dir, a_files[idx]))
            aud_t = audio_to_tensor(wav, sr)
            logits_a, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)
            logits_a = logits_a.to(device)
            probs_a = probs_a.to(device)

            # Image forward.
            lab_i, probs_i, logits_i, _ = extract_image_features(os.path.join(i_dir, i_files[idx]))
            logits_i = torch.tensor(logits_i, dtype=torch.float32).to(device)
            probs_i = torch.tensor(probs_i, dtype=torch.float32).to(device)

            # Normalize logits for gate.
            norm_logits_a = normalize_logits(logits_a, aud_logits_mean, aud_logits_std)
            norm_logits_i = normalize_logits(logits_i, img_logits_mean, img_logits_std)

            # Fuse.
            fused_probs, alpha = fusion_head.fuse_probs(
                probs_audio=probs_a,
                probs_image=probs_i,
                pre_softmax_audio=norm_logits_a,
                pre_softmax_image=norm_logits_i,
                return_gate=True
            )
            alpha_a, alpha_i = alpha[:, 0], alpha[:, 1]
            alpha_weights = torch.stack([alpha_a, alpha_i], dim=1)

            pred_f = class_names[torch.argmax(fused_probs).item()]

            # Bookkeeping.
            y_true.append(class_names.index(cname))
            y_a.append(class_names.index(pred_a))
            y_i.append(class_names.index(lab_i))
            y_f.append(class_names.index(pred_f))
            alpha_a_list.append(alpha_weights[0, 0].item())
            alpha_i_list.append(alpha_weights[0, 1].item())

            # Save all required columns (with fusion_mode and probabilities).
            records.append({
                "class": cname,
                "audio_file": a_files[idx],
                "image_file": i_files[idx],
                "audio_pred": pred_a,
                "image_pred": lab_i,
                "fusion_pred": pred_f,
                "fusion_mode": fusion_type,
                "audio_probs": probs_a.detach().cpu().numpy().tolist(),
                "image_probs": probs_i.detach().cpu().numpy().tolist(),
                "fusion_probs": fused_probs.detach().cpu().numpy().tolist()
            })

        print(f"[{cname}] paired {N} samples (audio {len(a_files)}/ image {len(i_files)})")

print("\n=== AUDIO vs IMAGE vs FUSION ===")
for name, y_hat in [("Audio", y_a), ("Image", y_i), ("Fusion", y_f)]:
    print(f"\n{name} report:")
    print(classification_report(y_true, y_hat, target_names=class_names, digits=3))

mean_a = np.mean(alpha_a_list) if alpha_a_list else 0
mean_i = np.mean(alpha_i_list) if alpha_i_list else 0
print(f"\nalpha audio (mean): {mean_a:.3f}   |   alpha image (mean): {mean_i:.3f}")

pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nResults written to {csv_path}\n")
