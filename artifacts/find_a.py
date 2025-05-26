import os
import sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
import torch
import torchaudio
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score
from utils import load_config
from experiments.ex_fusion.audio import load_audio_model, audio_to_tensor, audio_predict
from experiments.ex_fusion.image_model_interface import load_image_model, extract_image_features
from utils.AudioImageFusion_bi import AudioImageFusion

# Load config and set paths/classes.
config = load_config("config.yaml")
class_names = config["classes"]
audio_root = config["data"]["aud_train_dir"]
image_root = config["data"]["img_train_dir"]

audio_model_path = config["models"]["audio_model"]
image_model_path = config["models"]["image_model"]

out_dir = config["results_dir"]["root"]
os.makedirs(out_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

alphas_to_try = np.linspace(0.0, 1.0, 21)   # 0.0, 0.05, ..., 1.0

audio_model, _ = load_audio_model(model_path=audio_model_path)
image_model = load_image_model(model_path=image_model_path)
print("Audio and image backbones loaded.")

best_acc = -1
best_alpha = None
all_results = []

for alpha in alphas_to_try:
    print("\n" + "="*48)
    print(f"=== Testing alpha = {alpha:.2f} (audio weight, image={1-alpha:.2f}) ===")
    fusion_head = AudioImageFusion(
        num_classes=len(class_names),
        fusion_mode="avg",
        alpha=alpha
    ).to(device)

    records, y_true, y_a, y_i, y_f = [], [], [], [], []

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

            print(f"[{cname}] paired {N} samples (audio {len(a_files)}/ image {len(i_files)})")

            for idx in range(N):
                # Audio forward
                wav, sr = torchaudio.load(os.path.join(a_dir, a_files[idx]))
                aud_t = audio_to_tensor(wav, sr)
                logits_a, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)

                # Image forward
                lab_i, probs_i, logits_i, _ = extract_image_features(os.path.join(i_dir, i_files[idx]))
                logits_i = torch.tensor(logits_i, dtype=torch.float32).to(device)
                probs_i = torch.tensor(probs_i, dtype=torch.float32).to(device)
                logits_a = logits_a.to(device)
                probs_a = probs_a.to(device)

                # AVG fusion
                fused_probs = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i
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
                })

    acc = accuracy_score(y_true, y_f)
    all_results.append({
        "alpha": alpha,
        "fusion_acc": acc,
    })
    print(f"Fusion accuracy: {acc:.4f}")
    print(classification_report(y_true, y_f, target_names=class_names, digits=3))

    per_alpha_csv = os.path.join(out_dir, f"unsync_compare_avg_logits_alpha_{alpha:.2f}.csv")
    pd.DataFrame(records).to_csv(per_alpha_csv, index=False)
    print(f"Saved: {per_alpha_csv}")

    if acc > best_acc:
        best_acc = acc
        best_alpha = alpha

df_summary = pd.DataFrame(all_results)
df_summary.to_csv(os.path.join(out_dir, "avg_alpha_grid_search_summary.csv"), index=False)
print("\n==== GRID SEARCH SUMMARY ====")
print(df_summary)

print(f"\nBest alpha: {best_alpha:.2f} | Best fusion accuracy: {best_acc:.4f}")
