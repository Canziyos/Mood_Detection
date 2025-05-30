import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from utils.utils import load_config
from audio import load_audio_model, spectrogram_image_to_tensor,audio_predict, load_quant_aud_model
from image_model_interface import load_image_model, extract_image_features
from AudioImageFusion import AudioImageFusion

config = load_config("config.yaml")
class_names = config["classes"]

# Choose model type.
use_quantized = False

if use_quantized:
    audio_model_path = config["models"]["aud_quant_model"]
    image_model_path = config["models"]["img_quant_model"]
    audio_model, _ = load_quant_aud_model(model_path=audio_model_path)
    image_model = load_image_model(model_path=image_model_path, quantized=True)
    device = torch.device("cpu")
else:
    audio_model_path = config["models"]["audio_model"]
    image_model_path = config["models"]["image_model"]
    audio_model, device = load_audio_model(model_path=audio_model_path)
    image_model = load_image_model(model_path=image_model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

audio_root = config["data"]["aud_test_dir"]
image_root = config["data"]["img_test_dir"]
exp_name = "our_regular"
alpha = 0.6

out_dir = config["results_dir"]["root"]
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/fusion_{exp_name}_avg_alpha{alpha}.csv"

print("Audio and image backbones loaded.")
fusion_head = AudioImageFusion(num_classes=len(class_names), alpha=alpha).to(device)
print(f"Fusion loaded (alpha={alpha}).")

records, y_true, y_a, y_i, y_f = [], [], [], [], []

with torch.no_grad():
    for cname in class_names:
        a_dir = os.path.join(audio_root, cname)
        i_dir = os.path.join(image_root, cname)
        if not os.path.exists(a_dir) or not os.path.exists(i_dir):
            print(f"Skipping class '{cname}' (missing folder).")
            continue

        a_files = sorted([f for f in os.listdir(a_dir) if f.endswith(".png")])
        i_files = sorted([f for f in os.listdir(i_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        N = min(len(a_files), len(i_files))
        if N == 0:
            print(f"Skipping class '{cname}' (no paired files).")
            continue

        for idx in range(N):
            aud_t = spectrogram_image_to_tensor(os.path.join(a_dir, a_files[idx]))
            _, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)
            probs_a = probs_a.to(device)

            if use_quantized:
                lab_i, probs_i, _, _ = extract_image_features(os.path.join(i_dir, i_files[idx]), quantized=True)
            else:
                lab_i, probs_i, _, _ = extract_image_features(os.path.join(i_dir, i_files[idx]))
            probs_i = torch.tensor(probs_i, dtype=torch.float32).to(device)

            fused_probs = fusion_head.fuse_probs(
                probs_audio=probs_a,
                probs_image=probs_i,
            )

            pred_f = class_names[torch.argmax(fused_probs).item()]

            y_true.append(class_names.index(cname))
            y_a.append(class_names.index(pred_a))
            y_i.append(class_names.index(lab_i))
            y_f.append(class_names.index(pred_f))

            records.append({
                "class": cname,
                "audio_file": a_files[idx],
                "image_file": i_files[idx],
                "audio_pred": pred_a,
                "image_pred": lab_i,
                "fusion_pred": pred_f,
                "fusion_mode": "avg",
                "audio_probs": probs_a.detach().cpu().numpy().tolist(),
                "image_probs": probs_i.detach().cpu().numpy().tolist(),
                "fusion_probs": fused_probs.detach().cpu().numpy().tolist()
            })

        print(f"[{cname}] paired {N} samples (audio {len(a_files)}/ image {len(i_files)})")

print("\n=== AUDIO vs IMAGE vs FUSION ===")
for name, y_hat in [("Audio", y_a), ("Image", y_i), ("Fusion", y_f)]:
    print(f"\n{name} report:")
    print(classification_report(y_true, y_hat, target_names=class_names, digits=3))

pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nResults written to {csv_path}\n")
