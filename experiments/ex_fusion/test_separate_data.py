# --------------------------------------------------------------------------------#
# Unsynchronized audio/image fusion test (LOGITS-ONLY, supports "avg" and "gate") #
# --------------------------------------------------------------------------------#
import sys, os, torch, torchaudio, numpy as np, pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

# Add project root to sys.path for local imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from audio import load_audio_model, audio_to_tensor, audio_predict
from image_model_interface import load_image_model, extract_image_features
from src.fusion.AudioImageFusion import FusionAV

# Config-
class_names   = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
audio_root    = "./dataset/audio/EmoDB_audio_test"
image_root    = "./dataset/images/test"

fusion_type = "gate"   # "avg" or "gate".
alpha = 0.3   # Only used if fusion_type="avg".

ckpt_path = "./models/best_gate_head_logits.pth"
out_dir  = "./results"
os.makedirs(out_dir, exist_ok=True)
csv_path = f"{out_dir}/unsync_compare_{fusion_type}_logits.csv"
device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load base models.
audio_model, _ = load_audio_model(model_path="./models/mobilenetv2_aud.pth")
image_model = load_image_model(model_path="./models/mobilenetv2_img.pth")
print("Audio & image backbones loaded.")

# Load fusion model according to mode:
if fusion_type == "avg":
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="avg",
        alpha=alpha
    ).to(device)
    print(f"FusionAV loaded for avg fusion (alpha={alpha}).")
elif fusion_type == "gate":
    ckpt = torch.load(ckpt_path, map_location="cpu")
    fusion_head = FusionAV(
        num_classes=len(class_names),
        fusion_mode="gate"
    ).to(device)
    fusion_head.load_state_dict(ckpt["state_dict"])
    fusion_head.eval()
    print("Gate head loaded.")
else:
    raise ValueError(f"Unknown fusion_type: {fusion_type}")

# Inference loop.
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
            # audio forward.
            wav, sr = torchaudio.load(os.path.join(a_dir, a_files[idx]))
            aud_t = audio_to_tensor(wav, sr)
            logits_a, probs_a, _, pred_a = audio_predict(audio_model, aud_t, device)

            # image forward.
            lab_i, probs_i, logits_i, _ = extract_image_features(os.path.join(i_dir, i_files[idx]))
            logits_i = torch.tensor(logits_i, dtype=torch.float32).to(device)
            probs_i  = torch.tensor(probs_i,  dtype=torch.float32).to(device)
            logits_a = logits_a.to(device)
            probs_a  = probs_a.to(device)

            # fuse (using FusionAV in both modes).
            if fusion_type == "avg":
                fused_probs = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i
                )
                alpha_weights = torch.tensor([[fusion_head.alpha, 1-fusion_head.alpha]])
            elif fusion_type == "gate":
                fused_probs, alpha = fusion_head.fuse_probs(
                    probs_audio=probs_a,
                    probs_image=probs_i,
                    pre_softmax_audio=logits_a,
                    pre_softmax_image=logits_i,
                    return_gate=True
                )
                alpha_a, alpha_i = alpha[:, 0], alpha[:, 1]
                alpha_weights = torch.stack([alpha_a, alpha_i], dim=1)
            else:
                raise ValueError(f"Unknown fusion_type: {fusion_type}")

            pred_f = class_names[torch.argmax(fused_probs).item()]

            # Bookkeeping.
            y_true.append(class_names.index(cname))
            y_a.append(class_names.index(pred_a))
            y_i.append(class_names.index(lab_i))
            y_f.append(class_names.index(pred_f))
            alpha_a_list.append(alpha_weights[0, 0].item())
            alpha_i_list.append(alpha_weights[0, 1].item())

            records.append({
                "class": cname,
                "audio_file": a_files[idx],
                "image_file": i_files[idx],
                "audio_pred": pred_a,
                "image_pred": lab_i,
                "fusion_pred": pred_f,
                "alpha_audio": alpha_weights[0, 0].item(),
                "alpha_image": alpha_weights[0, 1].item(),
            })

        print(f"[{cname}] paired {N} samples (audio {len(a_files)}/ image {len(i_files)})")

# Metrics.
print("\n=== AUDIO vs IMAGE vs FUSION ===")
for name, y_hat in [("Audio", y_a), ("Image", y_i), ("Fusion", y_f)]:
    print(f"\n{name} report:")
    print(classification_report(y_true, y_hat, target_names=class_names, digits=3))

mean_a = np.mean(alpha_a_list) if alpha_a_list else 0
mean_i = np.mean(alpha_i_list) if alpha_i_list else 0
print(f"\nalfa audio (mean): {mean_a:.3f}   |   alfa image (mean): {mean_i:.3f}")

# Save csv.
pd.DataFrame(records).to_csv(csv_path, index=False)
print(f"\nResults written to {csv_path}")
